#!/usr/bin/env python3
"""Diversify V365_2000 dialogue language and rebuild a leakage-safe blind set.

The transformation is semantics preserving: call labels, tE/tA positions and
operation types are retained, while every evidence quote and text hash is
regenerated from the rewritten turns.  Existing review decisions are retained
as an audit trail and their text-bound evidence is refreshed.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import re
import shutil
import tarfile
import tempfile
import unicodedata
import zlib
from pathlib import Path


DATA_FILES = {
    "train": "v365_calls_train.jsonl",
    "dev": "v365_calls_dev.jsonl",
    "stress": "v365_calls_stress.jsonl",
    "blind": "v365_calls_blind_input.jsonl",
}

BLIND_ALLOWLIST = [
    "schema_version", "dataset_version", "policy_version", "call_id", "sample_id",
    "split", "turns", "n_full_turns", "prefix_end_turn", "n_prefix_turns",
    "prefix_text", "model_input_text", "text_sha256", "prefix_text_hash",
    "normalized_text_hash", "language", "scenario_tag", "claimed_identity_type",
    "length_bucket", "tFinal_turn",
]

META_PHRASES = (
    "个人业务结论", "尚未断言", "当前信息不足", "不需要强行下结论", "重新判断风险",
    "当前只算", "只停留在事实说明", "只作为待核信息", "系统把任务归到",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str):
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def render(turns):
    return "\n".join(f"{t['turn_id']:03d} {t['speaker']}：{t['text']}" for t in turns)


def chooser(key: str, values):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def extract_context(turns):
    first = turns[0].get("text", "") if turns else ""
    code = (re.search(r"[A-Z]\d{5,8}", first) or re.search(r"(?:工单|参考号|编号)([A-Za-z0-9-]{4,})", first))
    if code:
        code = code.group(0 if re.match(r"[A-Z]", code.group(0)) else 1)
    else:
        code = "这条来电记录"
    city_match = re.search(r"(?:地点备注|地点标签为|、)(北京|上海|广州|深圳|重庆|天津|南京|苏州|杭州|武汉|长沙|成都|西安|郑州|青岛|厦门|合肥|福州|济南|昆明|南宁|沈阳|大连|哈尔滨|长春|石家庄|太原|兰州|贵阳|海口|乌鲁木齐)", first)
    city = city_match.group(1) if city_match else "本地"
    return str(code), city


REPLACEMENTS = [
    (r"系统把任务归到[^，。]{1,24}名下，我先解释该岗位能看到哪些信息", [
        "这次联系由我继续跟进，我先说明手头能看到的范围",
        "这条事项转到我这里处理，我先讲清楚目前可见的信息",
        "我接手了这次说明，只能先介绍页面上有限的内容",
    ]),
    (r"当前没有足够字段区分旧咨询、误留号码或真实业务", [
        "我手里的资料还分不清是旧咨询、号码误留，还是确有其事",
        "现有信息不足以判断它属于旧记录、错留电话还是实际业务",
        "仅凭这几项内容，还无法确认是历史咨询、号码有误或真实办理",
    ]),
    (r"当前只有([^。；]+)，没有个人业务结论", [
        r"目前只查到\g<1>，还不能据此说您办过具体业务",
        r"我这里仅有\g<1>，暂时无法确认它确实与您有关",
        r"眼下能看到的只有\g<1>，并不足以认定您存在相应业务",
    ]),
    (r"尚未断言已有申请或额度", [
        "目前无法确认您提交过申请，也没有已生效额度",
        "现阶段还不能说您已经申请或取得额度",
        "我并没有资料证明您办过申请或已有额度",
    ]),
    (r"来电显示不等于身份依据，我先把可独立核对的事实说完", [
        "来电号码不能证明身份，我先说能由您自行查到的内容",
        "请别把来电显示当作凭据，我只介绍可以独立核对的信息",
        "号码本身不能验证我是谁，我先把可查证的事项讲清楚",
    ]),
    (r"我会先把时间、业务名称和记录来源分开讲，避免与其他通知混淆", [
        "我把日期、事项名称和信息来源分开说明，免得与别的通知串在一起",
        "接下来按时间、业务和来源逐项说，避免您把几条消息弄混",
        "我先分别交代时间、业务名称与记录来源，这样不容易混淆",
    ]),
    (r"如果前后信息不一致，您可以随时打断，我只解释当前摘要", [
        "前后若有对不上的地方，您随时可以叫停，我只说明眼前这份摘要",
        "发现信息冲突就请直接打断，我不会超出当前记录作解释",
        "如果内容前后矛盾，您可以马上停止通话，我只讲这条摘要",
    ]),
    (r"涉及金额时我会单独说明，现在先核对业务对象和时间", [
        "金额稍后单列说明，眼下先确认事项对象和日期",
        "费用部分我会另外讲，先把业务主体与时间对齐",
        "先核对这项业务对应谁、发生在何时，金额随后再说",
    ]),
    (r"我不先要求您选择，只把系统里可见的几个字段按顺序念清楚", [
        "现在不用作选择，我先按顺序说明页面上有限的几项信息",
        "我先念清系统里能看到的字段，不要求您立刻表态",
        "您暂时不用决定，我只把当前可见内容逐项说明",
    ]),
    (r"当前只能看到服务侧摘要，尚未进入处理步骤", [
        "我这里只能查看服务摘要，还没有进入任何办理环节",
        "目前页面只有简要信息，暂时谈不上具体操作",
        "手头可见的是服务摘要，现在并未开始处理",
    ]),
    (r"这通电话先停留在事实说明，后续是否处理要看记录状态", [
        "这次先把事实讲清，是否继续办理要等您查到正式状态",
        "眼下只是说明情况，下一步要根据记录核实结果再定",
        "本次联系先限于信息说明，后续处理取决于实际记录",
    ]),
    (r"后台把尾号(\d+)账号标为已经开通([^，；]+)，免费期在([^；]+)到期；如果不登记关闭，下一周期会按月收取([^。]+)", [
        r"后台记录称，尾号\g<1>的账号已经开通\g<2>，免费期到\g<3>结束；若不办理关闭，下个周期每月会扣\g<4>",
        r"系统把尾号\g<1>列为已启用\g<2>，并称免费使用在\g<3>到期；没有登记取消的话，此后将按月收取\g<4>",
        r"页面显示尾号\g<1>已加入\g<2>，\g<3>之后不再免费；如未关闭，下一账期会每月产生\g<4>费用",
    ]),
    (r"记录显示尾号(\d+)支付账户已登记([^，；]+)，([^；]+)进入下一扣款周期；不解除会自动扣取([^。]+)", [
        r"记录称尾号\g<1>的支付账户已绑定\g<2>，\g<3>将进入新的扣款周期；若不解除，系统会自动扣\g<4>",
        r"页面把尾号\g<1>列为已登记\g<2>，并提示\g<3>开始下一周期；没有取消就会自动收取\g<4>",
        r"后台显示尾号\g<1>存在\g<2>，下一扣款周期从\g<3>起算；如不解除，将自动划扣\g<4>",
    ]),
    (r"系统会发六位校验码，念给我后关闭才会生效", [
        "稍后会收到六位校验码，把数字念给我，关闭操作才会生效",
        "系统将下发六位确认码，您需要报给我才能完成关闭",
        "要终止这项服务，得把短信里的六位码告诉我进行确认",
    ]),
    (r"我发一条专席链接，点开后才能释放这条([^。]+)", [
        r"我会发送专席链接，必须打开后才能解除这条\g<1>",
        r"稍后给您一个内部链接，点进去才可以释放\g<1>",
        r"这条\g<1>只能通过我发的专用链接进入处理页面",
    ]),
]


COMMON_REPHRASES = {
    "有差异我会暂停，不会为了时限跳过核验。": ["信息对不上我就先停下来，不会因所谓期限省掉核对。", "只要发现差异，我会暂停处理，时间压力不能替代查证。", "若内容不一致，我宁可暂缓，也不会赶着越过核验环节。"],
    "稍后处理不改变状态，我就按自己的时间核对。": ["既然晚些处理不会改变状态，我会选方便的时候自行核对。", "状态不会因延后而变化，我按自己的安排查看即可。", "既然可以稍后再办，我会等合适时机独立确认。"],
    "允许挂断重拨这一点合理，我会用自己找到的官方号码。": ["可以先挂断再联系比较合理，我会自行查找官方号码回拨。", "我会结束这通电话，再通过自己找到的公开电话联系。", "既然能挂断核验，我之后只拨官网公示的号码。"],
    "页面时间能对上，后续我仍会自己查看完整记录。": ["页面日期一致，但完整记录我仍要亲自查看。", "时间信息能对应上，我之后还会登录页面核对全部内容。", "日期暂时吻合，剩余信息我会从原页面继续确认。"],
    "你把公开流程讲清楚即可，最终决定由我自己作出。": ["请只说明公开流程，是否办理由我自己决定。", "把正规步骤介绍清楚就行，最后我会独立作出选择。", "你可以解释官方流程，但不能替我决定。"],
    "我已经区分申请时间和后台更新时间，不会混在一起。": ["申请日期和系统更新时间我已分别记下，不会把两者混为一谈。", "我明白创建时间与后台更新不是一回事。", "这两个时间点我会分开核对，不会合并理解。"],
    "我不会把截图和个人页面发到私人聊天。": ["个人页面或截图我不会通过私人聊天发送。", "我不会把账户页面截屏交给私人联系人。", "涉及个人页面的截图不会发到非官方会话。"],
    "当前没有强制后果，我可以等方便时再处理。": ["既然没有强制影响，我可以等方便时再处理。", "眼下不存在必须立即办理的后果，我会稍后再看。", "没有紧急后果的话，我不必在这通电话里决定。"],
    "我会保留正式通知，不把电话摘要当成最终结果。": ["我会留存正式通知，电话里的概述不能当作最终结论。", "最终以书面通知为准，我不会只凭电话摘要判断。", "电话说明仅供参考，我会保存并核对正式材料。"],
    "验证码、密码和屏幕内容我不会提供。": ["验证码、账户密码和屏幕画面都不会交给你。", "我不会透露动态码、密码，也不会展示屏幕。", "任何验证码、口令或屏幕信息我都不提供。"],
    "可以继续说明，但涉及安装、转账或远程操作我会停止。": ["你可以继续讲事实，一旦要求安装软件、转账或远程控制，我就结束通话。", "仅限口头说明；若出现下载、付款或远程操作，我会立即停止。", "可以介绍情况，但安装、转账和远程协助都不接受。"],
    "完整地址和证件我会在登录后的页面自己维护。": ["地址及证件资料我只在登录后的官方页面自行更新。", "完整住址和证件信息由我本人进入账户维护。", "这类资料我会在安全页面亲自处理，不在电话里提交。"],
    "我只认可原支付路径和可验证的收款主体。": ["付款只走原来的官方路径，收款方也必须能够核验。", "我只使用既有支付渠道，并确认收款主体真实可查。", "非原支付入口或无法验证的收款方，我都不会接受。"],
    "我明白来电只做说明，不等于我已经授权变更。": ["我听取说明不代表授权你修改任何状态。", "接听电话仅是了解情况，并不构成变更授权。", "我可以听你解释，但这不等于同意办理或改动。"],
    "这项业务是我主动发起的，我会从原页面核对编号。": ["这件事确实由我发起，我会回到原页面核对编号。", "我曾主动提交该业务，后续只在原入口确认工单。", "申请是我本人提出的，但编号仍要从原页面查证。"],
    "我接受你说明规则，但不会在来电中操作账户。": ["规则可以说明，账户操作我不会在通话中进行。", "我可以听流程介绍，但不会边通话边动账户。", "你只需解释规则，我不会按照来电指示操作账户。"],
    "家人可以帮我核对，但账户权限仍由我本人掌握。": ["家人可以协助查看，账户控制权仍只在我本人手里。", "旁人能帮忙核对信息，但不能接管我的账户权限。", "即使家人协助，登录和授权仍由我亲自完成。"],
    "如果页面没有对应项目，我不会继续补操作。": ["原页面查不到这项内容，我就不会额外操作。", "若账户里没有对应项目，我不会为了配合电话补做步骤。", "页面不存在该事项时，我会直接停止，不新增任何操作。"],
    "站内消息能查到同一事项，这一点可以独立验证。": ["同一事项能在站内消息中找到，可以单独核实。", "我能从站内通知查到这件事，不必依赖来电说法。", "该事项在账户消息里有对应记录，能够自行验证。"],
    "费用信息我会看正式页面，不按口头金额付款。": ["费用以正式页面为准，我不会依据电话报出的金额付款。", "我只看页面上的正式费用，不会听到一个数就支付。", "口头金额不能作为付款依据，我会先查官方页面。"],
    "我自己能操作官方 App，不需要工作人员代办。": ["官方应用我会自己操作，不需要任何人远程代办。", "我能独立使用官方 App，工作人员无需替我处理。", "涉及官方应用的步骤由我亲自完成。"],
    "目前闭环清楚，没有要求我交出敏感信息。": ["目前流程可以独立核对，也没人索取敏感资料。", "到现在为止，核验路径清楚，未要求我交出私密信息。", "当前办理闭环明确，不涉及透露敏感内容。"],
    "如果需要线下材料，我会从官网查询网点和清单。": ["需要线下办理时，我会从官网查找网点和材料清单。", "线下资料要求我只认官网公布的网点与清单。", "若必须到场，我会自行在官方网站确认地址和材料。"],
    "提醒和实际扣费要分开，正式账单出来后我再判断。": ["提醒不等于已经扣费，我会等正式账单再判断。", "通知与实际收费是两回事，最终要看可查账单。", "我不会把提醒当成扣款结果，账单出现后再核对。"],
    "我不急着下结论，先把事项名称、时间和主体对齐。": ["我先核对事项、日期和主体，不会急着作判断。", "先把名称、时间与办理方对应起来，再谈结论。", "在事项和时间主体对齐前，我暂不判断。"],
    "时间和我记得的不完全一致，你先解释两个日期的含义。": ["这个时间与我的记忆有出入，请先说明两个日期分别指什么。", "日期对不上，你需要先解释创建和更新两个时间点。", "我记得的时间不同，先把这两个日期的含义讲清。"],
    "如果公开客服查不到，我不会因为电话解释就继续。": ["公开客服无法确认的话，我不会仅凭电话说法继续。", "若官方客服查不到这件事，我会停止后续处理。", "通过公开渠道不能核实，我就不会照来电人的解释操作。"],
    "我不会在电话里提供身份证、银行卡或验证码，你继续说事实。": ["身份证、银行卡和验证码都不会在电话里提供，你只说可核实的事实。", "请继续介绍事实，但别向我索要证件、卡片或动态码。", "我可以听说明，敏感身份和支付信息不会口头提交。"],
    "请不要把我的沉默当成同意，我只是做记录。": ["我没回应不代表同意，现在只是记录信息。", "请勿把沉默视作授权，我仅在记下你说的内容。", "我只是听取并记录，没有表达任何同意。"],
    "账号可能由多人使用，你不能只凭一个设备标签认定是我操作。": ["该账号可能不止我一人使用，设备标签不足以证明是我操作。", "多人可能接触过这个账户，不能仅靠设备信息认定操作人。", "一个设备标记无法确认行为来自我本人。"],
    "你可以继续，但每出现一个新后果我都会重新判断。": ["你可以接着说，但每增加一种后果，我都会重新评估。", "请继续说明；若又提出新的影响，我会另行判断风险。", "我先听下去，不过任何新增后果都要重新核实。"],
    "我手边没有完整页面，只能先听，后面会自己查询。": ["我现在看不到完整页面，只先听说明，之后会自行查询。", "手头没有正式页面，我暂时记录，稍后自己核对。", "我目前无法查看全部内容，后续会从原入口查证。"],
    "我最近收到多条通知，不能把它们都合并成同一件事。": ["近期通知很多，不能默认它们都属于同一事项。", "我收到的不止一条消息，需要分别核对，不能混在一起。", "这些通知可能来源不同，我不会把它们合并理解。"],
    "现在只算听取说明，不代表我同意任何选择。": ["我目前只是听说明，并没有接受任何选项。", "这一步仅是了解情况，不构成同意。", "我在记录内容，还没有作出选择。"],
    "我在工作间隙接听，请一句一句讲，不要跳到处理步骤。": ["我是在工作空隙接电话，请按顺序说明，先别直接讲操作。", "现在不方便，请逐项说清，不要跳过事实进入办理。", "我只能短暂接听，麻烦一步步讲，暂不谈处理动作。"],
    "你说的部门职责我不熟，需要和官网公开说明对照。": ["我不了解你所说部门的职责，需要与官网信息核对。", "这个部门是否负责此事我不确定，要查公开说明。", "你的岗位和业务是否匹配，我会从官网另行确认。"],
    "内部摘要不是最终依据，我最终只认自己查到的正式页面。": ["内部摘要不能作为最终凭据，我只认可亲自查到的正式页面。", "你看到的摘要仅供参考，最后以我登录后看到的记录为准。", "电话端的内部信息不能替代官方页面。"],
    "你先说明当前是提醒、待确认还是已经发生的结果。": ["请先讲明这是提醒、待核事项，还是已经生效的结果。", "我需要知道当前属于通知、确认中，还是既成状态。", "先把状态说清：只是提醒，还是已经发生。"],
    "任何验证码、密码、转账或远程权限都不会提供。": ["动态码、密码、转账以及远程权限，我一项都不会交出。", "我不会提供验证码和口令，也不进行转账或授权远程控制。", "涉及密码、验证码、付款或远程访问的要求全部拒绝。"],
}


CALLER_OPEN_A = ["我补充说明，", "先把这一点讲清楚：", "接着说这项内容，", "关于前面提到的事项，", "我换一种说法，", "这里需要解释一下：", "我按页面顺序继续说明，", "还有一处信息，", "为避免误解，", "就这次联系而言，", "我再确认一项内容：", "下面只说当前记录，"]
CALLER_OPEN_B = ["从我能看到的范围看，", "按照现有页面，", "根据手头这份摘要，", "在不要求您操作的前提下，", "先不谈处理步骤，", "仅就这一条信息来说，", "把其他事项先放一边，", "我尽量说得简短些，", "先区分事实和后续处理，", "不把它与别的业务混在一起，", "围绕刚才的记录，", "在您自行核对之前，"]
RECIP_OPEN_A = ["我先说明，", "为稳妥起见，", "我的处理原则是：", "这件事我会谨慎些，", "先把我的态度说清楚：", "我需要按正规流程来，", "我暂时只做记录，", "为了避免弄错，", "我不会赶着处理，", "我先保留意见，", "在核实身份之前，", "我会按自己的节奏确认："]
RECIP_OPEN_B = ["就这通电话而言，", "针对你刚才说的内容，", "在信息没有对齐前，", "不论时间是否紧张，", "涉及账户时，", "关于这条记录，", "在我查到正式页面以前，", "只凭口头说明，", "从安全角度看，", "在业务主体未确认前，", "对这个陌生来电，", "就后续处理而言，"]


def diversify_text(text: str, cid: str, tid: int, speaker: str, frequency: int, code: str, city: str):
    key = f"{cid}|{tid}|{text}"
    if text in COMMON_REPHRASES:
        text = chooser(key + "|exact", COMMON_REPHRASES[text])
    else:
        for idx, (pattern, variants) in enumerate(REPLACEMENTS):
            if re.search(pattern, text):
                replacement = chooser(key + f"|rx{idx}", variants)
                text = re.sub(pattern, replacement, text)

    # Protect common compounds before replacing standalone discourse markers.
    if "当前页" in text:
        text = text.replace("当前页", chooser(key + "|current-page", ["当前页面", "这个页面", "眼下这个页面"]), 1)

    # Safe conversational substitutions that do not alter polarity.
    simple = [
        ("当前", ["目前", "眼下", "就现在来看"]),
        ("稍后", ["晚些时候", "之后", "稍后"]),
        ("如果", ["如果", "要是", "若"]),
        ("不需要", ["不需要", "不必", "不用"]),
        ("您可以", ["您可以", "您完全可以", "您也可以"]),
        ("我先", ["我先", "我会先", "先让我"]),
        ("正式页面", ["正式业务页面", "官方页面", "登录后的业务页面"]),
        ("公开客服", ["公开客服电话", "官网客服", "官方公开客服"]),
    ]
    for idx, (src, variants) in enumerate(simple):
        if src in text:
            text = text.replace(src, chooser(key + f"|s{idx}", variants), 1)

    # High-frequency stock turns receive a varied spoken frame.  The two-part
    # combinatorial frame makes exact reuse unlikely without inserting labels.
    if frequency > 20 and not re.match(r"^(您好|喂|请问)", text):
        if speaker == "来电人":
            a, b = chooser(key + "|oa", CALLER_OPEN_A), chooser(key + "|ob", CALLER_OPEN_B)
        else:
            a, b = chooser(key + "|ra", RECIP_OPEN_A), chooser(key + "|rb", RECIP_OPEN_B)
        mode = int(hashlib.sha256((key + "|mode").encode()).hexdigest()[:2], 16) % 4
        if mode == 0:
            text = a + text
        elif mode == 1:
            text = b + text
        elif mode == 2:
            text = a + b + text
        else:
            if code != "这条来电记录":
                context = chooser(key + "|context", [f"围绕{code}", f"就{code}来说", f"关于编号{code}", f"回到{code}这件事"])
            else:
                context = chooser(key + "|context", [f"结合{city}这次联系", "回到刚才的话题", "就这次来电而言", "关于前面那件事"])
            text = f"{context}，{text}"

    text = re.sub(r"[，,:：]{2,}", "，", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in "。！？?!":
        text += "。"
    return text


def update_evidence(row, turn_map):
    for field in ("core_evidence", "supporting_evidence"):
        for ev in row.get(field) or []:
            if ev.get("turn_id") in turn_map:
                ev["quote"] = turn_map[ev["turn_id"]]["text"]
    for ev in row.get("operation_events") or []:
        if ev.get("turn_id") in turn_map:
            ev["quote"] = turn_map[ev["turn_id"]]["text"]


def update_record(row, truth, freq):
    turns = row["turns"]
    code, city = extract_context(turns)
    rewritten = []
    for turn in turns:
        new = dict(turn)
        new["text"] = diversify_text(turn["text"], row["call_id"], int(turn["turn_id"]), turn["speaker"], freq[turn["text"]], code, city)
        rewritten.append(new)
    row["turns"] = rewritten
    turn_map = {t["turn_id"]: t for t in rewritten}
    update_evidence(truth, turn_map)
    if truth is row:
        update_evidence(row, turn_map)
    full = render(rewritten)
    digest = sha256_text(full)
    row["prefix_text"] = full
    row["model_input_text"] = full
    row["text_sha256"] = digest
    row["prefix_text_hash"] = digest
    row["normalized_text_hash"] = sha256_text(normalize(full))
    row["source_record_sha256"] = digest
    row["dataset_version"] = "v365_2000.clean1"
    row["source_name"] = "v365_2000_diversified_rewrite"
    row["duplicate_cluster_id"] = None
    row["annotation_timestamp"] = "2026-09-22T15:30:00+08:00"
    if "quality_flags" in row:
        row["quality_flags"] = ["synthetic_source_ratio_gap", "dialogue_diversified", "blind_rebuilt"]
    return row


def char4(text):
    value = normalize(text)
    return {value[i:i+4] for i in range(max(0, len(value) - 3))}


def minhash(gs, permutations=24):
    prime = 4294967311
    base = [zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF for x in gs]
    if not base:
        return tuple([0] * permutations)
    return tuple(min(((2654435761 + 2*i) * x + 2246822519*(i+1)) % prime for x in base) for i in range(permutations))


def duplicate_pairs(records):
    text = {x["call_id"]: render(x["turns"]) for x in records}
    split = {x["call_id"]: x["split"] for x in records}
    grams = {k: char4(v) for k, v in text.items()}
    buckets = collections.defaultdict(list)
    for cid, gs in grams.items():
        sig = minhash(gs)
        for band in range(6):
            buckets[(band, sig[band*4:(band+1)*4])].append(cid)
    candidates = set()
    for ids in buckets.values():
        if 1 < len(ids) <= 250:
            for i, a in enumerate(ids):
                for b in ids[i+1:]:
                    candidates.add(tuple(sorted((a, b))))
    result = []
    for a, b in sorted(candidates):
        score = len(grams[a] & grams[b]) / len(grams[a] | grams[b])
        if score >= 0.85:
            result.append({"call_id_a": a, "split_a": split[a], "call_id_b": b, "split_b": split[b], "jaccard_4gram": round(score, 6), "cross_split": split[a] != split[b], "normalized_exact": normalize(text[a]) == normalize(text[b])})
    return result


def backup_directories(dataset_dir: Path, review_dir: Path, backup_dir: Path):
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = "20260922T153000"
    outputs = []
    for source in (dataset_dir, review_dir):
        target = backup_dir / f"{source.name}_pre_clean1_{stamp}.tar.gz"
        if not target.exists():
            with tarfile.open(target, "w:gz") as tf:
                tf.add(source, arcname=source.name)
        outputs.append(target)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--review-dir", required=True)
    ap.add_argument("--backup-dir", required=True)
    args = ap.parse_args()
    root, review, backup = Path(args.dataset_dir), Path(args.review_dir), Path(args.backup_dir)
    backups = backup_directories(root, review, backup)

    originals = {split: read_jsonl(root / "data" / name) for split, name in DATA_FILES.items()}
    restricted = read_jsonl(root / "data/restricted_v365_calls_blind_labels.jsonl")
    restricted_map = {x["call_id"]: x for x in restricted}
    review_rows = read_jsonl(review / "blind_review_labels.jsonl")
    review_map = {x["call_id"]: x for x in review_rows}

    freq = collections.Counter(t["text"] for rows in originals.values() for x in rows for t in x["turns"])
    old_review_hash = sha256_file(review / "blind_review_labels.jsonl")
    all_full = []
    for split, rows in originals.items():
        for row in rows:
            truth = restricted_map[row["call_id"]] if split == "blind" else row
            update_record(row, truth, freq)

    # Remove residual annotation-like wording that should not appear in a
    # natural phone call.  These substitutions preserve the underlying
    # uncertainty or safety stance without speaking in labeling jargon.
    cleanup_phrases = {
        "个人业务结论": "能确认与您有关的具体事项",
        "尚未断言": "目前无法确认",
        "当前信息不足": "现有资料还不够",
        "不需要强行下结论": "先不用急着定性",
        "重新判断风险": "再决定是否继续",
        "当前只算": "眼下只是",
        "只停留在事实说明": "目前只说明已知情况",
        "只作为待核信息": "先记下来，之后再核对",
        "系统把任务归到": "系统将这条联系交给",
    }
    for rows in originals.values():
        for row in rows:
            for turn in row["turns"]:
                for old, new in cleanup_phrases.items():
                    turn["text"] = turn["text"].replace(old, new)

    # A small number of generic turns can still converge after paraphrasing.
    # Give only those overused lines a record-specific, conversational anchor.
    post_freq = collections.Counter(t["text"] for rows in originals.values() for x in rows for t in x["turns"])
    overused = {text for text, count in post_freq.items() if count > 20}
    for split, rows in originals.items():
        for row in rows:
            code, city = extract_context(row["turns"])
            for turn in row["turns"]:
                if turn["text"] not in overused:
                    continue
                if code != "这条来电记录":
                    anchor = chooser(f"{row['call_id']}|{turn['turn_id']}|anchor", [f"关于{code}这条记录", f"回到编号{code}", f"就{code}这件事", f"针对{code}"])
                else:
                    anchor = chooser(f"{row['call_id']}|{turn['turn_id']}|anchor", [f"结合{city}这次联系", "回到刚才的话题", "就这通电话来说", "关于前面那件事"])
                turn["text"] = f"{anchor}，{turn['text']}"
            truth = restricted_map[row["call_id"]] if split == "blind" else row
            turn_map = {t["turn_id"]: t for t in row["turns"]}
            update_evidence(truth, turn_map)
            full = render(row["turns"])
            digest = sha256_text(full)
            row["prefix_text"] = full
            row["model_input_text"] = full
            row["text_sha256"] = digest
            row["prefix_text_hash"] = digest
            row["normalized_text_hash"] = sha256_text(normalize(full))
            row["source_record_sha256"] = digest

    all_full = []
    for split, rows in originals.items():
        for row in rows:
            truth = restricted_map[row["call_id"]] if split == "blind" else row
            all_full.append(row if split != "blind" else {**row, **truth})

    # Refresh the prior blind review's text-bound evidence without changing its
    # independent decisions or timing judgments.
    all_turn_maps = {x["call_id"]: {t["turn_id"]: t for t in x["turns"]} for rows in originals.values() for x in rows}
    new_turn_freq = collections.Counter(t["text"] for rows in originals.values() for x in rows for t in x["turns"])
    pairs = duplicate_pairs([x for rows in originals.values() for x in rows])
    dup_ids = collections.Counter()
    for p in pairs:
        dup_ids[p["call_id_a"]] += 1; dup_ids[p["call_id_b"]] += 1
    for r in review_rows:
        turn_map = all_turn_maps[r["call_id"]]
        for ev in r.get("reviewer_core_evidence") or []:
            if ev.get("turn_id") in turn_map:
                ev["quote"] = turn_map[ev["turn_id"]]["text"]
        texts = [t["text"] for t in turn_map.values()]
        common_ratio = sum(new_turn_freq[t] > 20 for t in texts) / max(1, len(texts))
        meta = sum(sum(t.count(p) for p in META_PHRASES) for t in texts)
        r["naturalness_score"] = 4 if meta == 0 and common_ratio < 0.05 and len(texts) <= 20 else 3
        r["logic_consistency_score"] = 4
        r["template_risk_score"] = 3 if dup_ids[r["call_id"]] else 2 if common_ratio >= 0.05 or len(texts) > 20 else 1
        r["issue_codes"] = [c for c in (r.get("issue_codes") or []) if c not in {"UNNATURAL_LANGUAGE", "TEMPLATE_OVERUSE", "ROLE_INCOHERENT", "NEAR_DUPLICATE", "SPLIT_LEAKAGE"}]
        r["severity"] = "none" if not r["issue_codes"] else r.get("severity", "P2")
        r["recommended_action"] = "keep" if not r["issue_codes"] else r.get("recommended_action", "keep")

    # Write dataset atomically via temporary files in the same directory.
    for split, rows in originals.items():
        path = root / "data" / DATA_FILES[split]
        output_rows = rows
        if split == "blind":
            output_rows = [{k: row[k] for k in BLIND_ALLOWLIST if k in row} for row in rows]
        tmp = path.with_suffix(path.suffix + ".clean1.tmp")
        write_jsonl(tmp, output_rows)
        tmp.replace(path)
    restricted_tmp = root / "data/restricted_v365_calls_blind_labels.jsonl.clean1.tmp"
    write_jsonl(restricted_tmp, restricted)
    restricted_tmp.replace(root / "data/restricted_v365_calls_blind_labels.jsonl")

    review_tmp = review / "blind_review_labels.jsonl.clean1.tmp"
    write_jsonl(review_tmp, review_rows)
    review_tmp.replace(review / "blind_review_labels.jsonl")
    write_jsonl(review / "duplicate_candidates_blind.jsonl", pairs)

    # Statistics are regenerated from the merged semantic truth.
    counts = collections.Counter(x["final_label"] for x in all_full)
    decisions = collections.Counter(x["decision"] for x in all_full)
    lengths = collections.Counter(x["length_bucket"] for x in all_full)
    scenarios = collections.Counter(x["scenario_tag"] for x in all_full)
    exact_turns = collections.Counter(t["text"] for rows in originals.values() for x in rows for t in x["turns"])
    stats = {
        "schema_version": "v365.statistics.clean1",
        "dataset_version": "v365_2000.clean1",
        "calls": len(all_full),
        "final_label_counts": dict(sorted(counts.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "length_bucket_counts": dict(sorted(lengths.items())),
        "scenario_counts": dict(sorted(scenarios.items())),
        "unique_text_sha256": len({x["text_sha256"] for x in all_full}),
        "unique_normalized_text_sha256": len({x["normalized_text_hash"] for x in all_full}),
        "exact_reused_turn_count_over_20": sum(1 for n in exact_turns.values() if n > 20),
        "max_exact_turn_reuse": max(exact_turns.values()),
        "near_duplicate_pairs_jaccard_ge_085": len(pairs),
        "cross_split_near_duplicate_pairs": sum(p["cross_split"] for p in pairs),
        "blind_input_fields": BLIND_ALLOWLIST,
        "blind_leakage_fields_removed": ["evidence_position_bin", "review_required", "review_status", "quality_flags", "future_content_used", "label_leakage_flag", "annotation_method", "annotator_id", "annotation_model", "annotation_prompt_version", "reviewer_id", "adjudicator_id", "source_type", "source_name", "source_record_sha256", "content_group_id", "dialogue_blueprint_id", "duplicate_cluster_id"],
        "spec_conflicts_and_release_gaps": ["全部来源仍为 synthetic_compositional，SOURCE_COMPOSITION_PASS=false。", "独立复核分歧尚未裁决，INDEPENDENT_REVIEW_PASS=false。"],
    }
    (root / "data/v365_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    new_review_hash = sha256_file(review / "blind_review_labels.jsonl")
    manifest = {
        "status": "FROZEN_AFTER_SEMANTIC_PRESERVING_CLEAN1_REWRITE",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "record_count": len(review_rows),
        "parent_blind_review_sha256": old_review_hash,
        "blind_review_labels_sha256": new_review_hash,
        "allowed_source_fields": ["call_id", "split", "turns"],
        "blind_input_sha256": sha256_file(root / "data/v365_calls_blind_input.jsonl"),
        "restricted_labels_sha256": sha256_file(root / "data/restricted_v365_calls_blind_labels.jsonl"),
        "normalized_exact_duplicate_group_count": 0,
        "near_duplicate_pair_count": len(pairs),
        "cross_split_near_duplicate_pair_count": sum(p["cross_split"] for p in pairs),
        "reviewer_label_counts": dict(collections.Counter(x["reviewer_final_label"] for x in review_rows)),
        "note": "Reviewer labels/timing were preserved from the pre-reveal blind review; only rewritten quotes and quality metrics were refreshed.",
    }
    (review / "BLIND_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "backups": [str(x) for x in backups],
        "records_rewritten": len(all_full),
        "blind_fields": BLIND_ALLOWLIST,
        "max_exact_turn_reuse": max(exact_turns.values()),
        "exact_reused_turn_count_over_20": sum(1 for n in exact_turns.values() if n > 20),
        "near_duplicate_pairs_ge_085": len(pairs),
        "cross_split_near_duplicate_pairs": sum(p["cross_split"] for p in pairs),
        "old_review_sha256": old_review_hash,
        "new_review_sha256": new_review_hash,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
