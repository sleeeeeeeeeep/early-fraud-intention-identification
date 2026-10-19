from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
DATASET_VERSION = "v364_section8_demo_v1"
SCHEMA_VERSION = "v364.independent_call.v1"
POLICY_VERSION = "v364_client_strict_multiscenario_boundary_v1"
TIMESTAMP = "2026-09-22T10:00:00+08:00"

OPERATION_TYPES = (
    "transfer",
    "otp",
    "download",
    "screen_share",
    "remote_control",
    "bank_operation",
    "unofficial_link",
    "account_or_password",
    "identity_document",
    "cash_gold_giftcard_or_courier",
    "crypto_asset",
    "sim_or_esim_operation",
)

STAGE_TARGET = {"normal": 0, "ambiguous": 1, "early_fraud": 2, "explicit_fraud": 3}


def turns(*items: tuple[str, str]) -> list[dict]:
    return [
        {"turn_id": i, "speaker": speaker, "text": text}
        for i, (speaker, text) in enumerate(items, 1)
    ]


CALLS = [
    {
        "key": "platform_fraud",
        "pair": "platform_membership_guarantee",
        "scenario": "platform_membership_guarantee",
        "identity": "platform_customer_service",
        "decision": "early_fraud",
        "tS": 1,
        "tE": 5,
        "tA": None,
        "core": [(5, "fabricated_enrollment", "来电人断言账号已登记免费保障，并把到期与自动按月收费、关闭处理连成个体化链条。")],
        "support": [(1, "identity_anomaly", "来电人自称平台保障专席，为后续个体化业务状态断言建立身份。")],
        "ops": [],
        "rationale": "严格甲方边界：平台身份、个体化已登记状态、到期收费与关闭链条同时出现即为 early_fraud；无需等待操作要求。",
        "turns": turns(
            ("来电人", "您好，我是星潮短视频平台保障专席，今天回访一项账户权益状态。"),
            ("接电人", "我最近没办过新业务，你先说具体是什么。"),
            ("来电人", "这项内容和直播账号的基础保障有关，我先核对您是否仍在使用尾号四二的账号。"),
            ("接电人", "账号还在用，但这不能说明我开过什么保障。"),
            ("来电人", "后台显示该账号三个月前登记了免费百万保障，今晚试用到期；如果不选择关闭，明天会按每月六百八十元进入连续收费。"),
            ("接电人", "我没有印象，也没在页面看到过这笔费用。"),
            ("来电人", "用户端可能只显示权益有效，关闭状态要由人工登记，所以我才电话确认您继续还是关闭。"),
            ("接电人", "我不会在来电里选择，我会自己进平台查询。"),
            ("来电人", "可以先查，但今晚的状态节点不会因为您挂断自动暂停。"),
            ("接电人", "那我更要从公开客服核实，今天不在电话里处理。"),
        ),
    },
    {
        "key": "platform_normal",
        "pair": "platform_membership_guarantee",
        "scenario": "platform_membership_guarantee",
        "identity": "platform_customer_service",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户主动在官方 App 提交已知付费会员取消申请；来电只确认工单结果，允许独立回拨且不虚构自动开通或收费后果。",
        "turns": turns(
            ("来电人", "您好，这里是星潮平台官方客服，回访您今天上午在 App 内提交的会员取消工单，编号可在帮助中心看到。"),
            ("接电人", "是我提交的，页面显示本月服务到期后不再续费。"),
            ("来电人", "我们只确认页面是否已经显示‘自动续费已关闭’，不会在电话里替您登录或收验证码。"),
            ("接电人", "我现在看到了，状态和你说的一致。"),
            ("来电人", "本期是您上月主动购买的已支付服务，权益会用到本月二十八日，不会再产生下一周期订单。"),
            ("接电人", "如果以后有争议，我从哪里核验？"),
            ("来电人", "请挂断后从 App 帮助中心进入该工单，或拨官网公开客服；不用回拨当前号码。"),
            ("接电人", "明白，这次只确认我主动申请的取消结果。"),
        ),
    },
    {
        "key": "police_fraud",
        "pair": "public_security_case",
        "scenario": "public_security_case",
        "identity": "public_security_officer",
        "decision": "explicit_fraud",
        "tS": 1,
        "tE": 7,
        "tA": 9,
        "core": [(7, "loss_or_legal_threat", "来电人以刑事责任和冻结风险施压，并要求秘密接受电话调查，形成权威冒用、隔离与后果威胁链。")],
        "support": [(5, "authority_impersonation", "来电人声称用户银行卡进入涉案资金名单。")],
        "ops": [(9, "screen_share"), (11, "bank_operation")],
        "rationale": "公安身份、个体化涉案断言、冻结和刑责威胁、禁止线下核验已足够 early；随后共享屏幕与网银操作升级为 explicit_fraud。",
        "turns": turns(
            ("来电人", "这里是海州市公安局经侦支队，我姓杜，联系您核查一宗跨省洗钱案件。"),
            ("接电人", "公安为什么用普通手机号码联系我？"),
            ("来电人", "案件在异地协查阶段，号码由专案平台外呼，先确认您是否叫周岚。"),
            ("接电人", "我只确认姓周，不会在电话里报身份证。"),
            ("来电人", "系统显示您名下一张银行卡被列入涉案资金流转名单，昨晚已经申请临时冻结。"),
            ("接电人", "我会去附近派出所核实。"),
            ("来电人", "现在属于侦查保密期，您去派出所或告诉家人都会被视为泄密；两小时内不配合，冻结会扩大并追究您的刑事责任。"),
            ("接电人", "正式调查应该有书面手续，你先给我案号。"),
            ("来电人", "案号在远程笔录页面里，您打开会议软件共享整个手机屏幕，我才能把电子警官证和文书同步给您看。"),
            ("接电人", "我不会共享屏幕。"),
            ("来电人", "那就先登录手机银行，把各账户余额和最近三笔交易逐项打开，我们需要做资金清查。"),
            ("接电人", "这些要求不符合正常办案流程，我会挂断并拨公开电话。"),
        ),
    },
    {
        "key": "police_normal",
        "pair": "public_security_case",
        "scenario": "public_security_case",
        "identity": "anti_fraud_center",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "针对用户主动提交的可疑来电记录进行安全回访，只给防范建议和公开核验路径，不调查资产、不要求保密或远程操作。",
        "turns": turns(
            ("来电人", "您好，这里是市反诈中心宣传回访席，跟进您昨晚通过一一零登记的可疑来电记录。"),
            ("接电人", "我确实报过，对方当时让我共享屏幕。"),
            ("来电人", "今天只确认您是否已经停止联系、有没有泄露验证码或发生转账，不会让您重做那些操作。"),
            ("接电人", "我没有转账，也没有共享。"),
            ("来电人", "请继续保留聊天和来电记录。如需核验本次回访，可挂断后拨一一零或在国家反诈中心 App 查询咨询。"),
            ("接电人", "需要我提供银行卡明细吗？"),
            ("来电人", "不需要。若发现损失，请本人携带材料到就近公安机关；电话里不要报密码、验证码或账户余额。"),
            ("接电人", "明白，我会自己核对账户。"),
        ),
    },
    {
        "key": "bank_fraud",
        "pair": "bank_credit_card_risk",
        "scenario": "bank_credit_card_risk",
        "identity": "bank_risk_control",
        "decision": "early_fraud",
        "tS": 1,
        "tE": 9,
        "tA": None,
        "core": [(9, "false_consequence", "来电人虚构不接受电话保护登记就会冻结全部银行卡并影响征信的个体化后果。")],
        "support": [(5, "fabricated_enrollment", "来电人断言用户名下存在未申请的白金卡和附加保险。")],
        "ops": [],
        "rationale": "银行身份与个体化虚构卡片、保险及冻结征信后果形成完整链条，虽未提出具体操作也应判 early_fraud。",
        "turns": turns(
            ("来电人", "您好，我是广域银行信用卡风险控制中心，核对一项新卡异常。"),
            ("接电人", "我只有一张普通信用卡，没有申请新卡。"),
            ("来电人", "系统上周收到一份线上升级申请，地址留的是外省。"),
            ("接电人", "我没去过外省，会不会是信息录错？"),
            ("来电人", "记录显示您名下已经生成一张白金卡，并附带每月三百二十元的账户保险。"),
            ("接电人", "我 App 里看不到，应该直接从官方客服查。"),
            ("来电人", "新卡还在内部风险池，普通客服和 App 暂时查不到，只能由我们先解除。"),
            ("接电人", "那你们发正式站内通知。"),
            ("来电人", "若今天下午前不接受电话保护登记，系统会把您现有银行卡一并冻结，并把异常申请报送征信。"),
            ("接电人", "我不会根据陌生来电做登记，会自己联系银行。"),
            ("来电人", "一旦转入征信流程，网点也只能等我们回传结果。"),
            ("接电人", "我现在挂断，从卡背电话核验。"),
        ),
    },
    {
        "key": "bank_normal",
        "pair": "bank_credit_card_risk",
        "scenario": "bank_credit_card_risk",
        "identity": "bank_customer_service",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户已在银行官方 App 主动申请换卡；客服只说明可核验进度，不索取敏感信息，不制造冻结或征信威胁。",
        "turns": turns(
            ("来电人", "您好，这里是广域银行客服，回访您昨天在官方 App 提交的信用卡到期换卡申请。"),
            ("接电人", "是我申请的，App 显示卡片正在制作。"),
            ("来电人", "我们只确认邮寄城市是否仍为杭州，完整地址请您自己在 App 安全中心核对，不要在来电中口述。"),
            ("接电人", "城市没变，具体地址我稍后自己看。"),
            ("来电人", "可以。旧卡在新卡激活前仍正常使用，不会因本次回访被冻结。"),
            ("接电人", "如果我不确认，申请会取消吗？"),
            ("来电人", "不会。您可以挂断，从卡背电话重新呼入，或在 App 工单里继续处理。"),
            ("接电人", "那我从 App 查看，不在电话里办。"),
        ),
    },
    {
        "key": "ecommerce_fraud",
        "pair": "ecommerce_refund",
        "scenario": "ecommerce_refund",
        "identity": "ecommerce_after_sales",
        "decision": "explicit_fraud",
        "tS": 1,
        "tE": 5,
        "tA": 7,
        "core": [(5, "abnormal_process", "来电人声称正常订单页无法退款，必须进入金融认证通道，明显偏离平台原路退款流程。")],
        "support": [(3, "false_consequence", "来电人虚构商品导致账户被列入批量赔付名单。")],
        "ops": [(7, "unofficial_link"), (9, "otp")],
        "rationale": "个体化赔付断言与异常金融通道已排除合理正常解释；短链接和验证码要求构成 explicit_fraud。",
        "turns": turns(
            ("来电人", "您好，我是优选商城售后专员，处理您上月购买的电热水壶质量赔付。"),
            ("接电人", "我买过水壶，但没有申请退款。"),
            ("来电人", "该批次被检测出绝缘问题，您的账户已经进入三倍赔付名单。"),
            ("接电人", "订单页面没有任何通知。"),
            ("来电人", "这是批量召回，普通订单页无法发起，必须先进入合作金融机构的赔付认证通道。"),
            ("接电人", "退款为什么要经过金融机构？"),
            ("来电人", "我发一个以字母开头的短链接，您打开后选择银行卡，系统才能测试退款接口。"),
            ("接电人", "我不会点陌生链接。"),
            ("来电人", "页面随后会发六位动态码，您念给我，验证成功后三千元会原路到账。"),
            ("接电人", "正常退款不需要把验证码告诉客服，我会联系平台。"),
        ),
    },
    {
        "key": "ecommerce_normal",
        "pair": "ecommerce_refund",
        "scenario": "ecommerce_refund",
        "identity": "ecommerce_after_sales",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户已在订单页提交退款；客服说明原路退回和站内可核验进度，不转移到金融通道或索取验证码。",
        "turns": turns(
            ("来电人", "您好，这里是优选商城官方售后，回访您在订单页提交的水壶退货申请。"),
            ("接电人", "是我申请的，快递上午已经取走。"),
            ("来电人", "仓库签收后，货款会按原支付路径退回；进度可在订单详情的退款页查看。"),
            ("接电人", "需要我提供银行卡吗？"),
            ("来电人", "不需要，也不会让您提供验证码或先付款。若原支付方式失效，平台会在站内工单提示您主动联系。"),
            ("接电人", "预计几天到账？"),
            ("来电人", "仓库验收后一个到三个工作日，最终以订单页时间为准；您可挂断后从 App 客服核验。"),
            ("接电人", "好的，我按订单页查询。"),
        ),
    },
    {
        "key": "airline_fraud",
        "pair": "airline_rebooking",
        "scenario": "airline_rebooking",
        "identity": "airline_rebooking_agent",
        "decision": "explicit_fraud",
        "tS": 1,
        "tE": 7,
        "tA": 9,
        "core": [(7, "false_consequence", "来电人以座位锁定和赔偿失效为由制造极短时限，并声称官方 App 不同步。")],
        "support": [(5, "inconsistent_identity_or_process", "来电人把改签补偿导向所谓清算合作方处理。")],
        "ops": [(9, "unofficial_link"), (11, "bank_operation")],
        "rationale": "航司身份、个体化航班异常、官方渠道不可核验、限时失效与外部清算路径形成 early；外链及银行余额验证构成 explicit。",
        "turns": turns(
            ("来电人", "您好，我是华江航空改签中心，您今晚飞成都的航班因机械检查取消。"),
            ("接电人", "我 App 里显示正常起飞。"),
            ("来电人", "运行控制刚下发，旅客端更新会晚一些，我们提前安排改签和四百元补偿。"),
            ("接电人", "我可以自己拨航司客服。"),
            ("来电人", "公开客服只处理普通退改，本次补偿由清算合作方统一发放。"),
            ("接电人", "那应该也能在订单里看到。"),
            ("来电人", "当前座位只保留十五分钟，超过时间补偿和免费改签都会失效，官方 App 暂时不会显示。"),
            ("接电人", "你准备让我怎么处理？"),
            ("来电人", "请打开我短信里的‘旅客清算中心’链接，先选一张常用银行卡。"),
            ("接电人", "陌生链接我不点。"),
            ("来电人", "也可以打开手机银行，把余额页面停留十秒，系统会验证这张卡能否接收补偿。"),
            ("接电人", "这不是正常改签流程，我会直接联系航司。"),
        ),
    },
    {
        "key": "airline_normal",
        "pair": "airline_rebooking",
        "scenario": "airline_rebooking",
        "identity": "airline_customer_service",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "航班变动已在官方 App 与机场信息中可核验；改签由用户主动进入订单完成，无外链、银行卡或短时威胁。",
        "turns": turns(
            ("来电人", "您好，这里是华江航空客服，您今晚的航班起飞时间调整了四十分钟，变动已同步到官方 App。"),
            ("接电人", "我刚看到 App 通知，是同一航班。"),
            ("来电人", "如果新时间不合适，可在原订单选择一次免费改签或退款；电话里不需要提供银行卡和验证码。"),
            ("接电人", "我想改到明早，可以现在决定吗？"),
            ("来电人", "不用在来电中决定。请您挂断后从 App 原订单操作，或者拨官网公布的客服号码。"),
            ("接电人", "机场柜台也能办吗？"),
            ("来电人", "可以，携带购票证件到值机柜台办理；可选航班和差价都以官方系统展示为准。"),
            ("接电人", "明白，我自己从订单改。"),
        ),
    },
    {
        "key": "courier_fraud",
        "pair": "courier_customs_fee",
        "scenario": "courier_customs_fee",
        "identity": "courier_customs_service",
        "decision": "explicit_fraud",
        "tS": 1,
        "tE": 7,
        "tA": 9,
        "core": [(7, "loss_or_legal_threat", "来电人虚构包裹涉违禁品和连带责任，并阻止用户通过官方单号核验。")],
        "support": [(5, "authority_impersonation", "来电人以海关联合处置身份建立权威。")],
        "ops": [(9, "transfer")],
        "rationale": "快递/海关身份、个体化违禁包裹、阻断官方核验与法律责任威胁足够 early；向个人账户缴保证金升级 explicit。",
        "turns": turns(
            ("来电人", "您好，我是速达国际件异常处理专员，您名下有一件寄往境外的包裹被海关拦截。"),
            ("接电人", "我最近没有寄国际件。"),
            ("来电人", "面单使用了您的姓名和手机号，申报内容是保健品。"),
            ("接电人", "把运单号给我，我在官网查。"),
            ("来电人", "该件已转海关联合处置，普通官网会隐藏编号，需要由我们内部核销。"),
            ("接电人", "既然不是我寄的，我可以报警。"),
            ("来电人", "包裹里检测到违禁成分，若今天不完成无责申明，寄件责任会直接记到您名下，之后可能限制出境。"),
            ("接电人", "无责申明在哪里办？"),
            ("来电人", "先向核验专员的个人收款账户转一千二百元保证金，海关确认身份后原路退回。"),
            ("接电人", "海关不会收个人账户保证金，我会联系官方渠道。"),
        ),
    },
    {
        "key": "courier_normal",
        "pair": "courier_customs_fee",
        "scenario": "courier_customs_fee",
        "identity": "courier_customer_service",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户知晓的国际件出现申报补充，运单状态和海关缴税入口均可从官方 App 独立核验，且不收个人账户款项。",
        "turns": turns(
            ("来电人", "您好，这里是速达国际件客服，您昨天寄出的文件包缺少英文品名，运单尾号八一六二。"),
            ("接电人", "是我寄的，App 里也显示等待补充申报。"),
            ("来电人", "请您挂断后打开速达 App，在该运单的‘申报资料’中补充；电话里不用提供证件照片。"),
            ("接电人", "如果产生税费，怎么缴？"),
            ("来电人", "只有海关审核后才会在运单页生成官方缴税单，不会让您向个人账户转保证金。"),
            ("接电人", "我现在不方便操作。"),
            ("来电人", "可以稍后处理，截止时间以 App 提示为准；若有疑问可从官网查运单并拨公开客服。"),
            ("接电人", "好，我回去从 App 补资料。"),
        ),
    },
    {
        "key": "loan_fraud",
        "pair": "loan_credit_repair",
        "scenario": "loan_credit_repair",
        "identity": "credit_repair_consultant",
        "decision": "early_fraud",
        "tS": 1,
        "tE": 7,
        "tA": None,
        "core": [(7, "abnormal_process", "来电人宣称可绕过征信异议流程、当日删除逾期并保证贷款通过，属于明显异常金融服务承诺。")],
        "support": [(5, "emotional_manipulation", "来电人利用用户贷款需求强化征信焦虑。")],
        "ops": [],
        "rationale": "征信修复身份、个体化逾期断言、绕过官方程序和保证审批的异常链条已足够 early。",
        "turns": turns(
            ("来电人", "您好，我是信用优化中心顾问，看到您近期查询过经营贷款额度。"),
            ("接电人", "我只在银行 App 看过测算，没有委托其他机构。"),
            ("来电人", "查询记录被合作渠道同步过来，您的征信里有两条逾期会影响审批。"),
            ("接电人", "逾期是否存在，我会自己查征信报告。"),
            ("来电人", "普通异议要一个月，您本周要用资金的话肯定来不及。"),
            ("接电人", "你们具体能做什么？"),
            ("来电人", "我们有内部修复名额，可以绕过异议流程当天删除逾期，并保证八十万元经营贷通过。"),
            ("接电人", "没有机构能保证银行审批。"),
            ("来电人", "这是合作银行预留通道，普通客服不会承认，名额今晚收回。"),
            ("接电人", "我不会通过这种渠道处理征信。"),
        ),
    },
    {
        "key": "loan_normal",
        "pair": "loan_credit_repair",
        "scenario": "loan_credit_repair",
        "identity": "bank_loan_officer",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户主动在银行官方渠道申请；来电不保证额度、不收费、不代做征信修复，所有材料和结果均回到官方 App。",
        "turns": turns(
            ("来电人", "您好，我是城商银行小微贷款中心，回访您今天在官方 App 提交的经营贷申请。"),
            ("接电人", "是我提交的，目前页面显示等待补充流水。"),
            ("来电人", "请只在 App 申请页上传，电话里不要发送征信报告、身份证照片或网银信息。"),
            ("接电人", "你能告诉我能批多少吗？"),
            ("来电人", "不能承诺额度和结果，最终以银行审批页面为准；本行不会收费删除征信逾期。"),
            ("接电人", "如果我今天不补，会不会影响征信？"),
            ("来电人", "不会，只是本次申请暂缓。您可取消申请，或从 App 客服、网点重新核验。"),
            ("接电人", "了解，我从官方页面决定。"),
        ),
    },
    {
        "key": "leader_fraud",
        "pair": "leader_payment_request",
        "scenario": "leader_payment_request",
        "identity": "company_leader",
        "decision": "explicit_fraud",
        "tS": 1,
        "tE": 5,
        "tA": 7,
        "core": [(5, "isolation_or_secrecy", "来电人以领导身份要求绕过财务审批并对同事保密，形成身份、紧迫性和隔离组合。")],
        "support": [(3, "urgency", "来电人制造投标保证金十分钟内到账的紧迫情境。")],
        "ops": [(7, "transfer")],
        "rationale": "熟人/领导身份、保密、绕审批和紧迫资金事项在转账前已足够 early；指定私人账户转款构成 explicit。",
        "turns": turns(
            ("来电人", "小林，我是赵总，我临时换了号码，正在外地谈项目。"),
            ("接电人", "赵总平时不直接找我付款，你说什么事？"),
            ("来电人", "合作方要补一笔投标保证金，十分钟内必须到账，不然项目名额就没了。"),
            ("接电人", "我让财务经理确认一下。"),
            ("来电人", "这次报价还没公开，先不要告诉财务和其他同事，也别走审批系统，等我回去再补手续。"),
            ("接电人", "不走审批我不能付款。"),
            ("来电人", "先从你个人账户转十八万元到合作方王某的卡里，备注写借款，稍后公司再还你。"),
            ("接电人", "我会拨赵总原号码核实，不会转私人账户。"),
        ),
    },
    {
        "key": "leader_normal",
        "pair": "leader_payment_request",
        "scenario": "leader_payment_request",
        "identity": "company_manager",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "已知经理只提醒公开审批任务，明确不得个人垫付或绕流程，付款主体和审批链可在公司系统核验。",
        "turns": turns(
            ("来电人", "小林，我是项目部赵经理，提醒你看一下公司审批系统里的投标保证金流程。"),
            ("接电人", "我看到申请了，发起人是采购部。"),
            ("来电人", "先核合同编号和收款单位，必须等财务、法务都通过后由公司账户支付。"),
            ("接电人", "时间写着今天下班前，来得及吗？"),
            ("来电人", "来不及就申请延期，不能个人垫付，也不能转给私人账户。"),
            ("接电人", "我需要在电话里确认密码吗？"),
            ("来电人", "不需要。你从公司内网处理，有疑问拨通讯录里的原号码或当面确认。"),
            ("接电人", "好，我按审批系统走。"),
        ),
    },
    {
        "key": "romance_fraud",
        "pair": "online_dating_investment",
        "scenario": "online_dating_investment",
        "identity": "online_romantic_contact",
        "decision": "explicit_fraud",
        "tS": 3,
        "tE": 9,
        "tA": 11,
        "core": [(9, "emotional_manipulation", "来电人把感情承诺、保密要求和保证收益的内部投资机会绑定，形成稳定欺骗链。")],
        "support": [(7, "off_platform_diversion", "来电人要求避开正规证券软件使用朋友维护的平台。")],
        "ops": [(11, "crypto_asset"), (13, "transfer")],
        "rationale": "网络关系、内部机会、保证收益、保密和非正规平台足够 early；购买稳定币并转入指定地址升级 explicit。",
        "turns": turns(
            ("来电人", "你下班了吗？我们认识快两个月了，我想和你商量以后一起生活的事。"),
            ("接电人", "可以谈，但我不想把感情和钱混在一起。"),
            ("来电人", "我只是想让你了解我在做的跨境数字资产项目。"),
            ("接电人", "我没接触过，也不准备冒险。"),
            ("来电人", "项目由我舅舅所在的机构内部管理，普通人进不去，我可以给你家庭名额。"),
            ("接电人", "有没有公开牌照和风险说明？"),
            ("来电人", "公开证券软件查不到，要用朋友维护的专属平台，注册后每天都能看到收益。"),
            ("接电人", "查不到的平台我不会用。"),
            ("来电人", "我把你当结婚对象才带你赚钱，收益保证每天百分之二；这件事别告诉家人，他们不了解会阻止我们。"),
            ("接电人", "你越这样说我越不放心。"),
            ("来电人", "先在交易所买五万元稳定币，我教你提到专属平台的钱包地址。"),
            ("接电人", "我不会买。"),
            ("来电人", "那先转一万元到平台客服账户做激活，今晚过后家庭名额就失效。"),
            ("接电人", "不要再联系我谈投资，我会核验这个平台。"),
        ),
    },
    {
        "key": "investment_normal",
        "pair": "online_dating_investment",
        "scenario": "online_dating_investment",
        "identity": "licensed_securities_adviser",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户预约持牌机构公开讲座；来电只给风险揭示和官网核验方式，不承诺收益、不引流私人平台或要求转账。",
        "turns": turns(
            ("来电人", "您好，我是恒川证券投资者教育中心，确认您在官网预约的线上风险讲座。"),
            ("接电人", "是我预约的，主题是基金波动风险。"),
            ("来电人", "讲座链接会出现在证券 App 消息中心，工作人员不会通过私人聊天发送开户链接。"),
            ("接电人", "会推荐具体产品吗？"),
            ("来电人", "只做一般教育，不保证收益，也不会要求购买数字货币、向个人账户转款或共享交易屏幕。"),
            ("接电人", "如何确认讲师资质？"),
            ("来电人", "可在公司官网和证券业协会公开信息中核验；不方便参加可以直接取消。"),
            ("接电人", "好的，我从 App 进入。"),
        ),
    },
    {
        "key": "tax_fraud",
        "pair": "government_tax_review",
        "scenario": "government_tax_review",
        "identity": "tax_review_officer",
        "decision": "early_fraud",
        "tS": 1,
        "tE": 7,
        "tA": None,
        "core": [(7, "false_consequence", "来电人断言企业进入异常名录并将冻结发票、处罚，同时要求通过电话内部通道处理，形成权威和个体化后果链。")],
        "support": [(5, "blocks_verification", "来电人声称电子税务局和公开热线均无法查询该内部问题。")],
        "ops": [],
        "rationale": "政务税务身份、个体化异常名录、阻断官方核验与处罚后果组合达到 early，即使尚未要求转账。",
        "turns": turns(
            ("来电人", "您好，我是市税务稽核协办岗，联系贵公司处理零申报异常。"),
            ("接电人", "我是代理会计，电子税务局显示申报成功。"),
            ("来电人", "成功的是企业端提交，监管共享库没有收到负责人确认。"),
            ("接电人", "那应该能在电子税务局看到待办。"),
            ("来电人", "该问题属于内部风控标记，电子税务局和公开热线暂时都查不到。"),
            ("接电人", "查不到我不能按电话处理。"),
            ("来电人", "系统已把企业列入异常名录，今天不通过电话协办通道确认，明早会冻结发票并产生两万元处罚。"),
            ("接电人", "正式处罚应有文书和申辩渠道。"),
            ("来电人", "文书要完成内部确认后才能生成，现在只能先登记负责人态度。"),
            ("接电人", "我会去税务大厅核验，不接受电话内部通道。"),
        ),
    },
    {
        "key": "tax_normal",
        "pair": "government_tax_review",
        "scenario": "government_tax_review",
        "identity": "tax_service_hotline",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户已在电子税务局提交更正；热线仅解释可核验状态和公开办理路径，不虚构内部标记或即时处罚。",
        "turns": turns(
            ("来电人", "您好，这里是税务服务热线回访席，跟进您昨天在电子税务局提交的申报更正咨询。"),
            ("接电人", "是我提交的，工单里问经营地址怎么改。"),
            ("来电人", "处理意见已经写入工单：先在电子税务局查看登记信息，确需变更再通过公开入口提交。"),
            ("接电人", "今天不改会冻结发票吗？"),
            ("来电人", "不会因本次咨询立即冻结。任何处罚或限制都应有正式页面状态和法定文书。"),
            ("接电人", "你能替我远程改吗？"),
            ("来电人", "不能，也不会索要密码、验证码或共享屏幕。您可挂断后拨一二三六六核验工单。"),
            ("接电人", "好，我自己从电子税务局办理。"),
        ),
    },
    {
        "key": "telecom_fraud",
        "pair": "telecom_sim_esim",
        "scenario": "telecom_sim_esim",
        "identity": "telecom_security_service",
        "decision": "explicit_fraud",
        "tS": 1,
        "tE": 5,
        "tA": 7,
        "core": [(5, "false_consequence", "来电人虚构号码被异地注册 eSIM 且即将停机并影响银行短信的个体化紧急后果。")],
        "support": [(3, "fabricated_enrollment", "来电人断言用户名下手机号已在异地生成副卡配置。")],
        "ops": [(7, "sim_or_esim_operation"), (9, "otp")],
        "rationale": "运营商身份、个体化 eSIM 异常、官方渠道不可核验和停机后果达到 early；安装配置和提交验证码构成 explicit。",
        "turns": turns(
            ("来电人", "您好，我是通信运营商账户安全中心，核对您手机号的一次异地 eSIM 开通。"),
            ("接电人", "我没有申请 eSIM。"),
            ("来电人", "昨晚有设备在外省提交了副卡配置，号码尾号与您当前号码一致。"),
            ("接电人", "我可以去营业厅或者拨官方客服。"),
            ("来电人", "安全池记录不会同步给普通客服和营业厅，只有本专席能撤回；如果二十分钟内不处理，原实体卡会停机，银行短信也会转到新设备。"),
            ("接电人", "你说怎么撤回？"),
            ("来电人", "先安装我发来的运营商配置文件，进入 eSIM 管理页删除新设备。"),
            ("接电人", "我不会安装陌生配置。"),
            ("来电人", "系统还会给您发送一组六位校验码，念给我就能锁定号码。"),
            ("接电人", "验证码不能提供，我会用官网号码核验。"),
        ),
    },
    {
        "key": "telecom_normal",
        "pair": "telecom_sim_esim",
        "scenario": "telecom_sim_esim",
        "identity": "telecom_customer_service",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户主动在官方 App 预约补卡；来电只提醒线下核验和官方自助路径，不远程配置 eSIM 或索取验证码。",
        "turns": turns(
            ("来电人", "您好，这里是运营商客服，确认您今天在官方 App 预约的实体卡补换业务。"),
            ("接电人", "是我预约的，准备周末去营业厅。"),
            ("来电人", "请本人携带有效证件到预约营业厅办理，电话里不会收身份证照片或短信验证码。"),
            ("接电人", "旧卡现在还能用吗？"),
            ("来电人", "可以，只有您在营业厅完成新卡激活后旧卡才失效；本次回访不会远程停机。"),
            ("接电人", "能改成 eSIM 吗？"),
            ("来电人", "是否支持请在官方 App 或营业厅查询，不要安装他人发来的配置文件。"),
            ("接电人", "明白，我按预约到店。"),
        ),
    },
    {
        "key": "insurance_fraud",
        "pair": "insurance_policy_service",
        "scenario": "insurance_policy_service",
        "identity": "insurance_account_service",
        "decision": "early_fraud",
        "tS": 1,
        "tE": 7,
        "tA": None,
        "core": [(7, "fee_threat", "来电人断言用户被免费登记高额保障、试用到期自动扣费且只能由电话专席解除。")],
        "support": [(5, "fabricated_enrollment", "来电人声称保单由银行卡活动自动赠送并已登记在用户名下。")],
        "ops": [],
        "rationale": "保险身份、无依据自动登记、到期扣费和电话关闭链条满足甲方 early_fraud 边界。",
        "turns": turns(
            ("来电人", "您好，我是安济保险账户服务岗，确认一项高额医疗保障的续用状态。"),
            ("接电人", "我没有买过你们公司的保险。"),
            ("来电人", "这不是您主动购买的纸质保单，是合作银行活动附带的线上权益。"),
            ("接电人", "银行 App 和保险 App 都没有显示。"),
            ("来电人", "活动在去年刷卡时自动赠送，后台已经登记到您名下，客户端只显示基础权益。"),
            ("接电人", "自动赠送为什么要问续用？"),
            ("来电人", "免费期今晚结束，明天会从合作银行卡每月扣四百九十元；只有账户专席能登记关闭，普通客服看不到。"),
            ("接电人", "我不会在电话里关闭，会从银行和保险公司核验。"),
            ("来电人", "可以核验，但超过今晚系统会先生成首期账单。"),
            ("接电人", "我不接受这种说法。"),
        ),
    },
    {
        "key": "insurance_normal",
        "pair": "insurance_policy_service",
        "scenario": "insurance_policy_service",
        "identity": "insurance_customer_service",
        "decision": "normal",
        "tS": None,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [],
        "ops": [],
        "rationale": "用户主动投保且保单可在官方 App 核验；续期由用户自主选择，来电不虚构免费开通、强制扣费或专席关闭。",
        "turns": turns(
            ("来电人", "您好，这里是安济保险客服，回访您去年通过官方 App 购买的医疗险续期报价。"),
            ("接电人", "我知道这份保单，月底到期，App 里也能看到。"),
            ("来电人", "本次只提醒查看新年度保费和条款，不会因为来电自动续保。"),
            ("接电人", "我还没决定续不续。"),
            ("来电人", "可以到期前在 App 自主选择；若未授权续期，不会从银行卡扣款。"),
            ("接电人", "需要在电话里确认银行卡吗？"),
            ("来电人", "不需要。请从保单页或官网公开客服核验，工作人员不会索取验证码。"),
            ("接电人", "好，我看完条款再决定。"),
        ),
    },
    {
        "key": "platform_ambiguous",
        "pair": "platform_membership_guarantee",
        "scenario": "platform_membership_general_query",
        "identity": "platform_customer_service",
        "decision": "ambiguous",
        "tS": 1,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [(1, "identity_anomaly", "陌生来电自称平台会员客服，但只提出一般规则说明。")],
        "ops": [],
        "rationale": "只有平台身份和一般会员规则询问，没有断言用户已开通、到期、收费或待关闭，也没有异常操作。",
        "turns": turns(
            ("来电人", "您好，我是星潮平台会员客服，想做一次会员规则满意度回访。"),
            ("接电人", "你具体要问什么？"),
            ("来电人", "主要了解用户是否看得懂续费和取消页面，也可以只听规则说明。"),
            ("接电人", "你没有说我已经开通吧？"),
            ("来电人", "我现在只看到回访名单，无法确认您账户是否有会员，具体状态要由您自己查询。"),
            ("接电人", "那我先不继续，之后从 App 看。"),
        ),
    },
    {
        "key": "bank_ambiguous",
        "pair": "bank_credit_card_risk",
        "scenario": "bank_service_callback",
        "identity": "bank_customer_service",
        "decision": "ambiguous",
        "tS": 1,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [(1, "identity_anomaly", "陌生来电自称银行服务回访，但尚未给出个体化异常状态。")],
        "ops": [],
        "rationale": "身份暂不可核实，但只询问是否方便接听服务回访，未出现账户后果、异常流程或敏感操作。",
        "turns": turns(
            ("来电人", "您好，我是城商银行服务回访人员，请问现在方便听一分钟吗？"),
            ("接电人", "先说什么业务，我不会在电话里报账户信息。"),
            ("来电人", "目前只做客户服务体验调查，不涉及交易和账户变更。"),
            ("接电人", "你能提供工号让我从公开客服核验吗？"),
            ("来电人", "可以，我姓林，工号一七六，您也可以直接挂断后拨卡背电话。"),
        ),
    },
    {
        "key": "police_ambiguous",
        "pair": "public_security_case",
        "scenario": "public_security_incomplete_call",
        "identity": "public_security_officer",
        "decision": "ambiguous",
        "tS": 1,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [(1, "authority_impersonation", "陌生来电自称公安机关，但通话在说明具体事项前中断。")],
        "ops": [],
        "rationale": "权威身份可疑但文本残缺，没有个体化案件、威胁、阻断核验或高风险操作，不能强判诈骗。",
        "turns": turns(
            ("来电人", "您好，这里是海州市公安局综合服务台，有一项登记事项需要和您说明。"),
            ("接电人", "什么登记事项？请给我公开可核验的依据。"),
            ("来电人", "我先确认您是否方便记录部门名称。"),
            ("接电人", "信号不好，我没听清。"),
        ),
    },
    {
        "key": "courier_ambiguous",
        "pair": "courier_customs_fee",
        "scenario": "courier_delivery_exception",
        "identity": "courier_customer_service",
        "decision": "ambiguous",
        "tS": 1,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [(1, "identity_anomaly", "来电人自称快递客服并声称有异常件，但未说明异常类型或处理路径。")],
        "ops": [],
        "rationale": "存在个体化快件异常提醒，但正常派送异常同样合理；未出现收费、法律后果、外链或敏感信息要求。",
        "turns": turns(
            ("来电人", "您好，我是速达快递客服，系统里有一件关联您手机号的快件显示异常。"),
            ("接电人", "我最近有几个包裹，你说运单尾号。"),
            ("来电人", "我这里暂时只看到异常回访任务，完整运单信息还没有加载出来。"),
            ("接电人", "那我自己从快递 App 查询。"),
            ("来电人", "可以，当前不用提供验证码或支付任何费用。"),
        ),
    },
    {
        "key": "leader_ambiguous",
        "pair": "leader_payment_request",
        "scenario": "workplace_identity_incomplete",
        "identity": "company_leader",
        "decision": "ambiguous",
        "tS": 1,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [(1, "identity_anomaly", "陌生号码自称单位负责人，但未提出保密、绕流程或资金要求。")],
        "ops": [],
        "rationale": "身份尚未核实，事项也不完整；正常临时换号与身份欺骗均可能，证据不足。",
        "turns": turns(
            ("来电人", "小陈，我是项目部周主任，今天换了一个临时号码。"),
            ("接电人", "周主任原号码还能联系吗？你找我什么事？"),
            ("来电人", "是下周项目会的材料安排，我稍后把具体清单发到公司系统。"),
            ("接电人", "我先从通讯录原号码确认一下。"),
            ("来电人", "可以，确认后再看系统通知。"),
        ),
    },
    {
        "key": "investment_ambiguous",
        "pair": "online_dating_investment",
        "scenario": "investment_education_invitation",
        "identity": "investment_adviser",
        "decision": "ambiguous",
        "tS": 1,
        "tE": None,
        "tA": None,
        "core": [],
        "support": [(1, "identity_anomaly", "陌生来电自称投资顾问，邀请参加公开讲座，但未出现收益承诺或资金操作。")],
        "ops": [],
        "rationale": "投资主题本身不能决定诈骗；当前没有个体化资产断言、保证收益、私域引流或付款要求。",
        "turns": turns(
            ("来电人", "您好，我是金融知识讲座的课程顾问，邀请您参加周末的公开线上活动。"),
            ("接电人", "我没报名，你们从哪里拿到号码？"),
            ("来电人", "名单来自一次公开展会登记，但我现在无法核对具体展位。"),
            ("接电人", "活动会让我开户或买产品吗？"),
            ("来电人", "目前只介绍课程时间，没有要求开户或付款，您也可以拒绝。"),
            ("接电人", "那我先不参加。"),
        ),
    },
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def render_text(call_turns: list[dict]) -> str:
    return "\n".join(f"{turn['turn_id']:03d} {turn['speaker']}：{turn['text']}" for turn in call_turns)


def length_bucket(n_turns: int) -> str:
    if n_turns <= 8:
        return "2-8"
    if n_turns <= 20:
        return "9-20"
    if n_turns <= 40:
        return "21-40"
    return "41+"


def evidence_bin(t_e: int | None, n_turns: int) -> str | None:
    if t_e is None:
        return None
    ratio = t_e / n_turns
    if ratio <= 0.25:
        return "Q1"
    if ratio <= 0.50:
        return "Q2"
    if ratio <= 0.75:
        return "Q3"
    return "Q4"


def stage_trajectory(spec: dict) -> list[dict]:
    result = []
    for turn in spec["turns"]:
        turn_id = turn["turn_id"]
        if spec["decision"] == "normal":
            decision = "normal"
        elif spec["decision"] == "ambiguous":
            decision = "normal" if turn_id < spec["tS"] else "ambiguous"
        elif turn_id < spec["tS"]:
            decision = "normal"
        elif turn_id < spec["tE"]:
            decision = "ambiguous"
        elif spec["tA"] is not None and turn_id >= spec["tA"]:
            decision = "explicit_fraud"
        else:
            decision = "early_fraud"
        result.append({"prefix_end_turn": turn_id, "decision": decision, "stage_target": STAGE_TARGET[decision]})
    return result


def evidence_rows(spec: dict, key: str) -> list[dict]:
    by_id = {turn["turn_id"]: turn for turn in spec["turns"]}
    return [
        {
            "turn_id": turn_id,
            "quote": by_id[turn_id]["text"],
            "evidence_type": evidence_type,
            "explanation": explanation,
        }
        for turn_id, evidence_type, explanation in spec[key]
    ]


def operation_rows(spec: dict) -> tuple[dict, list[dict]]:
    by_id = {turn["turn_id"]: turn for turn in spec["turns"]}
    present = {name: False for name in OPERATION_TYPES}
    events = []
    for turn_id, operation_type in spec["ops"]:
        present[operation_type] = True
        events.append(
            {
                "turn_id": turn_id,
                "quote": by_id[turn_id]["text"],
                "operation_type": operation_type,
            }
        )
    return present, events


def build_record(index: int, spec: dict) -> dict:
    call_id = f"V364_DEMO_CALL_{index:04d}"
    sample_id = f"V364_DEMO_FULL_{index:04d}"
    full_text = render_text(spec["turns"])
    decision = spec["decision"]
    final_label = "fraud" if decision in {"early_fraud", "explicit_fraud"} else decision
    final_binary = 1 if final_label == "fraud" else 0 if final_label == "normal" else -1
    binary_target = 1 if final_binary == 1 else 0 if final_binary == 0 else -1
    operation_present, operation_events = operation_rows(spec)
    source_id = f"synthetic:{DATASET_VERSION}:{spec['key']}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "call_id": call_id,
        "sample_id": sample_id,
        "source_sample_id": source_id,
        "content_group_id": f"V364_GROUP_{spec['pair'].upper()}",
        "duplicate_cluster_id": call_id,
        "split": "demo",
        "source_type": "synthetic_authored",
        "source_name": "v364_section8_multiscenario_demo",
        "source_record_sha256": digest(json.dumps(spec["turns"], ensure_ascii=False, sort_keys=True)),
        "text_sha256": digest(full_text),
        "turns": spec["turns"],
        "n_full_turns": len(spec["turns"]),
        "prefix_end_turn": len(spec["turns"]),
        "n_prefix_turns": len(spec["turns"]),
        "prefix_text": full_text,
        "model_input_text": full_text,
        "prefix_text_hash": digest(full_text),
        "normalized_text_hash": digest(normalize(full_text)),
        "language": "zh-CN",
        "scenario_tag": spec["scenario"],
        "scenario_pair_id": spec["pair"],
        "claimed_identity_type": spec["identity"],
        "length_bucket": length_bucket(len(spec["turns"])),
        "evidence_position_bin": evidence_bin(spec["tE"], len(spec["turns"])),
        "final_label": final_label,
        "final_binary_label": final_binary,
        "decision": decision,
        "stage_target": STAGE_TARGET[decision],
        "binary_target": binary_target,
        "binary_loss_mask": 0 if decision == "ambiguous" else 1,
        "stage_loss_mask": 1,
        "sample_weight": 1.0,
        "prefix_relation": "clean_normal" if decision == "normal" else None,
        "tS_turn": spec["tS"],
        "tE_turn": spec["tE"],
        "tA_turn": spec["tA"],
        "tFinal_turn": len(spec["turns"]),
        "early_detectable": decision in {"early_fraud", "explicit_fraud"},
        "explicit_operation_present": operation_present,
        "operation_events": operation_events,
        "core_evidence": evidence_rows(spec, "core"),
        "supporting_evidence": evidence_rows(spec, "support"),
        "insufficient_reason": spec["rationale"] if decision == "ambiguous" else "",
        "boundary_rationale": spec["rationale"],
        "turn_stage_labels": stage_trajectory(spec),
        "annotation_method": "rule_guided_authored_demo_and_deterministic_self_check",
        "annotator_id": "codex_v364_demo_author",
        "annotation_model": "openai_codex_current",
        "annotation_prompt_version": POLICY_VERSION,
        "reviewer_id": "codex_v364_deterministic_review",
        "adjudicator_id": None,
        "confidence": "medium" if decision == "ambiguous" else "high",
        "review_required": False,
        "review_status": "single_pass_demo_validated",
        "disagreement_type": None,
        "future_content_used": False,
        "label_leakage_flag": False,
        "quality_flags": [],
        "annotation_timestamp": TIMESTAMP,
    }
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [build_record(index, spec) for index, spec in enumerate(CALLS, 1)]
    calls_path = OUT / "v364_calls_demo.jsonl"
    calls_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )

    decision_counts = Counter(record["decision"] for record in records)
    final_counts = Counter(record["final_label"] for record in records)
    scenario_counts = Counter(record["scenario_pair_id"] for record in records)
    length_counts = Counter(record["length_bucket"] for record in records)
    evidence_counts = Counter(
        record["evidence_position_bin"] for record in records if record["evidence_position_bin"] is not None
    )
    source_counts = Counter(record["source_type"] for record in records)
    paired = defaultdict(set)
    for record in records:
        paired[record["scenario_pair_id"]].add(record["final_label"])

    statistics = {
        "schema_version": "v364.section8_demo.statistics.v1",
        "dataset_version": DATASET_VERSION,
        "calls": len(records),
        "independent_call_ids": len({record["call_id"] for record in records}),
        "decision_counts": dict(sorted(decision_counts.items())),
        "final_label_counts": dict(sorted(final_counts.items())),
        "scenario_pair_counts": dict(sorted(scenario_counts.items())),
        "matched_fraud_normal_scenario_pairs": sum(
            1 for labels in paired.values() if {"fraud", "normal"}.issubset(labels)
        ),
        "length_bucket_counts": dict(sorted(length_counts.items())),
        "evidence_position_counts": dict(sorted(evidence_counts.items())),
        "source_type_counts": dict(sorted(source_counts.items())),
        "explicit_operation_calls": sum(record["decision"] == "explicit_fraud" for record in records),
        "early_without_explicit_operation_calls": sum(record["decision"] == "early_fraud" for record in records),
        "client_strict_boundary_applied_across_scenarios": True,
        "demo_limitations": [
            "本文件为第八节规范化示范，不计作 2000 通 MVP 已完成。",
            "全部为 synthetic_authored，未达到规范建议的真实或人工自然改写来源比例。",
            "示范仅使用 demo split，尚未建立 Train、Dev、Frozen Stress、Blind Test 冻结切分。",
            "示范长度以 2-20 轮为主，尚未达到 21-40 轮及 41+ 轮目标分布。",
            "尚未完成独立双标；review_status 仅表示规则和确定性校验完成。",
        ],
    }
    (OUT / "v364_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
