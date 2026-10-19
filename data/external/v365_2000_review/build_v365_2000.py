from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCHEMA_VERSION = "v365.independent_call.v1"
DATASET_VERSION = "v365_2000_candidate_v1"
POLICY_VERSION = "v365_client_strict_multiscenario_boundary_v1"
TIMESTAMP = "2026-09-22T12:00:00+08:00"
SEED = 3652000

SPLIT_QUOTAS = {
    "train": {"fraud": 600, "normal": 600, "ambiguous": 300},
    "dev": {"fraud": 80, "normal": 80, "ambiguous": 40},
    "stress": {"fraud": 60, "normal": 60, "ambiguous": 30},
    "blind": {"fraud": 60, "normal": 60, "ambiguous": 30},
}

CLASS_LENGTH_QUOTAS = {
    "fraud": {"2-8": 20, "9-20": 280, "21-40": 260, "41+": 240},
    "normal": {"2-8": 30, "9-20": 320, "21-40": 290, "41+": 160},
    "ambiguous": {"2-8": 250, "9-20": 100, "21-40": 50, "41+": 0},
}

TE_BIN_QUOTAS = {"Q1": 120, "Q2": 200, "Q3": 240, "Q4": 240}

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

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安", "苏州", "宁波", "合肥", "长沙", "青岛", "厦门", "福州", "南昌", "郑州", "太原", "昆明", "南宁", "石家庄", "济南"]
TIMES = ["上午九点左右", "临近中午", "下午一点多", "下午三点左右", "傍晚下班前", "晚饭后", "晚上九点前后"]
MONTH_DAYS = [f"{month}月{day}日" for month in range(1, 13) for day in (3, 7, 11, 16, 21, 26)]
PEOPLE = ["周先生", "陈女士", "林师傅", "王经理", "赵老师", "刘会计", "孙老板", "吴同学", "郑阿姨", "何先生"]
DEVICES = ["旧手机", "备用机", "平板", "店铺电脑", "收银设备", "家人手机"]
CHANNELS = ["官方 App", "官网", "公开客服电话", "线下柜台", "站内工单", "政务服务网"]
SCENARIO_CHANNELS = {
    "platform_membership": ["官方 App", "官网", "公开客服电话", "站内工单"],
    "million_guarantee": ["官方 App", "官网", "公开客服电话", "站内工单"],
    "automatic_debit": ["官方 App", "官网", "公开客服电话", "站内工单"],
    "trial_expiry": ["官方 App", "官网", "公开客服电话", "站内工单"],
    "public_security_regulator": ["公开客服电话", "线下窗口", "政务服务网"],
    "bank_credit": ["官方 App", "官网", "公开客服电话", "线下柜台"],
    "ecommerce_refund": ["官方 App", "官网", "公开客服电话", "站内工单"],
    "travel_rebooking": ["官方 App", "官网", "公开客服电话", "线下柜台"],
    "courier_customs": ["官方 App", "官网", "公开客服电话", "站内工单"],
    "loan_credit_repair": ["官方 App", "官网", "公开客服电话", "线下柜台"],
    "acquaintance_leader": ["公司内网", "原工作群", "办公系统", "原通讯录"],
    "dating_investment": ["官网", "公开客服电话", "线下营业部", "官方 App"],
    "government_tax": ["官网", "公开客服电话", "线下窗口", "政务服务网"],
    "telecom_sim": ["官方 App", "官网", "公开客服电话", "线下营业厅"],
    "insurance_claim": ["官方 App", "官网", "公开客服电话", "线下服务网点"],
    "recruitment_education": ["官网", "公开客服电话", "学校就业系统", "正规招聘平台"],
}
AMOUNTS = [68, 96, 128, 198, 260, 320, 498, 680, 980, 1200, 2400, 3600, 5800, 12000]


SCENARIOS = {
    "platform_membership": {
        "organizations": ["星潮短视频平台", "云播内容平台", "乐映直播平台"],
        "identities": ["会员服务专席", "直播权益客服", "账户服务回访员"],
        "topics": ["直播会员", "创作者会员", "商家高级会员"],
        "fraud_evidence": "后台把尾号{tail}账号标为已经开通{topic}，免费期在{date}到期；如果不登记关闭，下一周期会按月收取{amount}元。",
        "evidence_type": "fabricated_enrollment",
        "normal_context": "回访您刚才在{channel}主动提交的{topic}取消工单，编号{ref}可由您自行核验。",
        "normal_status": "该会员是您此前主动购买的，关闭后只使用到已付款周期结束，不会生成下一期订单。",
        "ambiguous": "本次只做{topic}规则满意度回访，我这里没有权限确认您是否开通过具体服务。",
        "ops": [("screen_share", "请共享手机屏幕，我带您进入隐藏的会员关闭入口。"), ("unofficial_link", "我发一条专席链接，点开后才能释放这条会员记录。"), ("otp", "系统会发六位校验码，念给我后关闭才会生效。")],
    },
    "million_guarantee": {
        "organizations": ["惠享平台", "安选账户中心", "优联支付平台"],
        "identities": ["保障权益客服", "百万保障专员", "账户保障回访岗"],
        "topics": ["百万保障", "账户安全保障", "支付风险保障"],
        "fraud_evidence": "系统显示您的账户三个月前登记了{topic}免费权益，{date}结束试用；未关闭会从绑定账户每月扣{amount}元。",
        "evidence_type": "fee_threat",
        "normal_context": "您在{channel}咨询过{topic}规则，本次只解释保障范围，不断言账户已开通。",
        "normal_status": "是否存在保障、是否收费都以您自己登录后的页面和协议为准，电话不会替您开通或关闭。",
        "ambiguous": "我负责说明{topic}的一般规则，当前名单不包含具体开通或扣费状态。",
        "ops": [("bank_operation", "请打开手机银行进入扣款协议页，我需要看您当前显示的授权状态。"), ("remote_control", "安装远程协助工具后由专席替您关闭保障。"), ("otp", "把银行发来的动态码告诉我，系统才能取消后续扣费。")],
    },
    "automatic_debit": {
        "organizations": ["快捷支付服务中心", "联付账户中心", "聚合支付平台"],
        "identities": ["扣款协议专员", "支付授权客服", "连续服务审核员"],
        "topics": ["连续扣款协议", "自动代扣服务", "周期支付授权"],
        "fraud_evidence": "记录显示尾号{tail}支付账户已登记{topic}，{date}进入下一扣款周期；不解除会自动扣取{amount}元。",
        "evidence_type": "fee_threat",
        "normal_context": "您在{channel}主动查询过尾号{tail}的{topic}，本次只回访查询结果。",
        "normal_status": "协议是否存在以及扣款主体均可在支付设置页查看，客服不会电话索取密码或代为解除。",
        "ambiguous": "本次是{topic}功能调查，只询问页面是否容易理解，不代表您的账户存在该协议。",
        "ops": [("account_or_password", "请把支付账户和登录口令报给我，我从后台解除协议。"), ("screen_share", "打开支付设置并共享屏幕，我帮您找到隐藏授权。"), ("unofficial_link", "通过我发的协议清理链接操作，普通页面看不到这条授权。")],
    },
    "trial_expiry": {
        "organizations": ["悦听会员中心", "云储存服务平台", "智能办公权益中心"],
        "identities": ["试用到期专员", "订阅服务客服", "权益续用回访员"],
        "topics": ["云空间试用", "办公套件体验版", "音视频高级权益"],
        "fraud_evidence": "后台显示您在{date}领取了{topic}，今天免费期结束；若不选择关闭，将按每月{amount}元自动续用。",
        "evidence_type": "fabricated_enrollment",
        "normal_context": "您主动在{channel}申请了{topic}，本次回访只核对已显示的到期日期。",
        "normal_status": "试用默认不续费，是否购买由您在官方页面主动确认，来电不会生成订单。",
        "ambiguous": "我只说明{topic}的试用规则，无法从当前回访名单判断您是否领取过。",
        "ops": [("unofficial_link", "请进入专席发送的续用管理页面，选择关闭并提交银行卡。"), ("download", "需要安装旧版权益管理应用，普通 App 无法取消。"), ("otp", "把短信确认码告诉我，我替您登记不续用。")],
    },
    "public_security_regulator": {
        "organizations": ["海州市公安局", "金融监管协查中心", "检察案件联络处"],
        "identities": ["经侦协查员", "案件核验专员", "监管调查人员"],
        "topics": ["涉案账户协查", "资金清查", "身份冒用案件"],
        "fraud_evidence": "协查系统显示您名下账户已进入{topic}名单，{date}前不接受保密调查就会扩大冻结并追究责任，且不得向家人或属地窗口说明。",
        "evidence_type": "loss_or_legal_threat",
        "normal_context": "跟进您通过公开渠道登记的可疑来电记录{ref}，只确认是否发生损失。",
        "normal_status": "工作人员不会电话调查资产、要求保密或远程操作；如需报案请本人到就近机关。",
        "ambiguous": "这里自称是综合服务台，有一项登记事项要说明，但目前尚未调取具体案件内容。",
        "ops": [("screen_share", "打开会议软件共享整部手机屏幕，我给您展示远程笔录。"), ("bank_operation", "现在登录手机银行逐笔打开余额和交易明细接受资金清查。"), ("transfer", "把资金转入指定安全账户，核验结束后会原路返还。")],
    },
    "bank_credit": {
        "organizations": ["广域银行", "城商银行", "安融消费金融"],
        "identities": ["风险控制专员", "信用卡审核员", "账户安全客服"],
        "topics": ["异常白金卡", "信用账户升级", "未确认贷款额度"],
        "fraud_evidence": "系统显示您名下已生成{topic}并附带{amount}元费用；{date}前不做电话保护登记，将冻结现有账户并报送征信。",
        "evidence_type": "false_consequence",
        "normal_context": "回访您在{channel}主动提交的银行卡换卡或贷款咨询工单{ref}。",
        "normal_status": "申请结果和费用只在银行正式页面展示，不保证额度，也不会因拒绝来电立即冻结或影响征信。",
        "ambiguous": "这是一项银行服务体验回访，目前不涉及具体账户、交易或信用状态。",
        "ops": [("otp", "把银行发送的动态码念给我，才能撤销异常申请。"), ("screen_share", "共享银行 App 屏幕，我核对风险控制页面。"), ("transfer", "先向风险验证账户转{amount}元，系统确认后自动退回。")],
    },
    "ecommerce_refund": {
        "organizations": ["优选商城", "惠购电商平台", "海淘订单中心"],
        "identities": ["售后赔付专员", "订单退款客服", "召回处理人员"],
        "topics": ["商品召回赔付", "订单异常退款", "质量问题理赔"],
        "fraud_evidence": "订单{ref}已经进入{topic}名单，但普通订单页无法处理，必须通过外部金融认证，否则赔付资格在{date}失效。",
        "evidence_type": "abnormal_process",
        "normal_context": "回访您在订单页主动提交的{topic}申请{ref}，退回进度可由您独立查询。",
        "normal_status": "退款只按原支付路径返回，不要求先缴费、提供验证码或改到陌生账户。",
        "ambiguous": "系统只显示一项售后回访任务，尚未加载具体订单和处理状态。",
        "ops": [("unofficial_link", "打开我发的外部赔付链接，先选择一张银行卡。"), ("otp", "把赔付页面发送的动态码告诉我，退款接口才能通过。"), ("transfer", "先支付{amount}元通道验证金，赔付款会连同验证金退回。")],
    },
    "travel_rebooking": {
        "organizations": ["华江航空", "远航票务中心", "城市酒店集团"],
        "identities": ["改签专员", "旅客服务客服", "订单调整人员"],
        "topics": ["航班取消补偿", "酒店订单退款", "行程改签赔付"],
        "fraud_evidence": "系统显示订单{ref}发生{topic}，官方 App 暂不显示；若不在十五分钟内走合作清算通道，免费改签和{amount}元补偿都会失效。",
        "evidence_type": "false_consequence",
        "normal_context": "您已从{channel}看到订单{ref}的行程变动，本次只说明自主退改路径。",
        "normal_status": "所有可选行程、差价和退款都在原订单或官方柜台完成，不通过私人链接处理。",
        "ambiguous": "来电只称有一项行程服务提醒，尚未提供可核对的订单、日期或变动内容。",
        "ops": [("unofficial_link", "点击短信里的旅客清算链接并选择收款银行卡。"), ("bank_operation", "打开手机银行余额页停留十秒，验证银行卡能否接收补偿。"), ("otp", "把航司验证码报给我，我替您锁定新航班。")],
    },
    "courier_customs": {
        "organizations": ["速达快递", "国际件申报中心", "口岸物流服务台"],
        "identities": ["异常件专员", "海关协办客服", "国际物流核验员"],
        "topics": ["违禁品包裹", "国际件申报异常", "实名寄件责任"],
        "fraud_evidence": "系统显示关联您手机号的包裹{ref}涉及{topic}，普通官网隐藏记录；{date}前不办无责核销将承担责任并受出境限制。",
        "evidence_type": "loss_or_legal_threat",
        "normal_context": "您已在{channel}看到运单{ref}等待补充申报，本次只提醒从原运单办理。",
        "normal_status": "税费只在官方运单生成，不向个人账户收保证金，也不在电话中接收证件照片。",
        "ambiguous": "来电称有一件关联手机号的快件异常，但完整运单和异常类型尚未显示。",
        "ops": [("transfer", "向核验人员的个人账户转{amount}元保证金，核销后退回。"), ("identity_document", "把身份证正反面拍照发到专员聊天账号完成无责申明。"), ("unofficial_link", "打开海关协办短链接填写银行卡和寄件信息。")],
    },
    "loan_credit_repair": {
        "organizations": ["信用优化中心", "融信贷款审核部", "企业融资顾问平台"],
        "identities": ["征信修复顾问", "贷款预审专员", "融资服务经理"],
        "topics": ["逾期记录修复", "经营贷预审", "内部额度激活"],
        "fraud_evidence": "内部名单显示您已进入{topic}通道，可绕过官方审核在当天删除逾期并保证贷款通过；名额在{date}失效。",
        "evidence_type": "abnormal_process",
        "normal_context": "回访您在银行{channel}主动提交的贷款申请{ref}，材料只由您在正式页面上传。",
        "normal_status": "银行不保证额度、不收费删除征信记录，审批和异议结果均由官方系统决定。",
        "ambiguous": "来电自称融资顾问，只询问是否愿意听一般贷款规则，尚未断言已有申请或额度。",
        "ops": [("transfer", "先缴{amount}元包装服务费到顾问账户，才能锁定额度。"), ("identity_document", "把完整征信报告和身份证照片发给我做内部包装。"), ("bank_operation", "登录网银下载流水并通过共享页面提交给审核端。")],
    },
    "acquaintance_leader": {
        "organizations": ["项目部", "采购中心", "老同事联络组"],
        "identities": ["单位负责人", "熟人同事", "合作方经理"],
        "topics": ["投标保证金", "临时材料款", "合同履约款"],
        "fraud_evidence": "负责人临时换了这个号码，这笔{topic}必须保密，需绕过财务审批并在{date}前由个人先行垫付{amount}元。",
        "evidence_type": "isolation_or_secrecy",
        "normal_context": "提醒查看公司系统中的{topic}审批{ref}，身份和事项均可从原通讯录与内网核验。",
        "normal_status": "必须等财务和法务审批后由公司账户付款，不得个人垫付或转给私人账户。",
        "ambiguous": "陌生号码自称单位负责人，只说稍后会在公司系统发送项目材料，未提出保密或资金要求。",
        "ops": [("transfer", "先从个人账户转{amount}元到合作方私人卡，备注借款。"), ("otp", "把付款动态码发给我，我替你在外地完成审批。"), ("cash_gold_giftcard_or_courier", "来不及转账就购买等值购物卡，把卡密发给我先应急。")],
    },
    "dating_investment": {
        "organizations": ["线上交友群", "数字资产学习社", "私人投资工作室"],
        "identities": ["网络朋友", "投资导师", "项目内部人员"],
        "topics": ["内部数字资产项目", "高收益量化账户", "跨境投资名额"],
        "fraud_evidence": "这个{topic}只对亲近的人开放，收益保证每天增长；我们的关系不要告诉家人，名额也不能外传。",
        "evidence_type": "emotional_manipulation",
        "normal_context": "回访您在持牌机构{channel}预约的投资者教育活动{ref}，只提供公开风险说明。",
        "normal_status": "不保证收益、不引流私人平台开户，也不要求购买数字资产或向个人账户付款。",
        "ambiguous": "这次只邀请您参加一般金融知识活动，来源和资质尚未核实，也未提出投资或付款。",
        "ops": [("crypto_asset", "先在交易所购买{amount}元稳定币，再提到我发的钱包地址。"), ("transfer", "向平台客服账户转{amount}元激活资金账户。"), ("unofficial_link", "下载朋友维护的投资页面并用邀请码注册，正规软件查不到。")],
    },
    "government_tax": {
        "organizations": ["税务稽核协办岗", "市场监管年报服务中心", "政务企业服务台"],
        "identities": ["税务核查员", "年报审核员", "政务协办人员"],
        "topics": ["零申报异常", "地址登记不一致", "年报待核验"],
        "fraud_evidence": "内部系统显示企业{ref}已因{topic}进入异常名录，公开系统和热线查不到；{date}前不走电话协办将冻结业务并处罚{amount}元。",
        "evidence_type": "false_consequence",
        "normal_context": "回访您在{channel}提交的政务咨询工单{ref}，只解释公开办理路径。",
        "normal_status": "任何限制或处罚都应有正式页面状态和法定文书，来电不代办、不收密码或验证码。",
        "ambiguous": "自称政务服务人员，只说有一项企业登记事项待说明，尚未给出具体异常或处理要求。",
        "ops": [("unofficial_link", "进入内部联审链接登录企业账户，普通政务网没有该入口。"), ("screen_share", "共享电子税务局屏幕，我远程指导解除异常。"), ("transfer", "先缴{amount}元协办保证金，处罚流程才会暂停。")],
    },
    "telecom_sim": {
        "organizations": ["通信运营商安全中心", "号码服务专席", "宽带账户审核部"],
        "identities": ["SIM 安全专员", "eSIM 核验客服", "通信账户管理员"],
        "topics": ["异地 eSIM 副卡", "号码异常停机", "宽带账户迁移"],
        "fraud_evidence": "安全中心记录显示号码尾号{tail}已异地生成{topic}，普通客服和营业厅看不到；二十分钟内不撤回将停机并转移银行短信。",
        "evidence_type": "false_consequence",
        "normal_context": "确认您在{channel}主动预约的补卡或套餐变更工单{ref}。",
        "normal_status": "只有本人在官方 App 或营业厅完成后状态才改变，客服不会远程停机、收验证码或安装配置。",
        "ambiguous": "来电自称运营商回访，只询问通信服务体验，尚未给出号码异常或变更状态。",
        "ops": [("sim_or_esim_operation", "安装我发送的运营商配置文件，再删除所谓异地 eSIM。"), ("otp", "把运营商发送的六位校验码念给我锁定号码。"), ("download", "下载专席版安全工具，普通营业厅应用无法撤回副卡。")],
    },
    "insurance_claim": {
        "organizations": ["安济保险", "惠民保障中心", "联合保险服务台"],
        "identities": ["保单服务客服", "保障续期专员", "理赔回访人员"],
        "topics": ["高额医疗保障", "账户附加保险", "免费意外保障"],
        "fraud_evidence": "系统显示{topic}由合作活动自动登记到您名下，免费期在{date}结束；之后每月扣{amount}元且只能由电话专席关闭。",
        "evidence_type": "fabricated_enrollment",
        "normal_context": "回访您在{channel}主动购买或申请理赔的保单{ref}，记录可由您自行查看。",
        "normal_status": "续保由用户在保单页主动选择，理赔按正式材料处理，不电话索取验证码或要求私人付款。",
        "ambiguous": "来电只介绍一般保险服务规则，当前无法确认您是否持有对应保单。",
        "ops": [("bank_operation", "打开手机银行扣款协议页，我核对并解除保单授权。"), ("otp", "把银行卡动态码告诉我，系统才能关闭续期。"), ("screen_share", "共享保单和银行页面，由专席同步取消。")],
    },
    "recruitment_education": {
        "organizations": ["在线招聘项目组", "职业培训服务中心", "校园助学咨询平台"],
        "identities": ["招聘专员", "课程顾问", "助学审核员"],
        "topics": ["兼职入职激活", "培训分期名额", "助学账户审核"],
        "fraud_evidence": "内部记录显示您已进入{topic}名单并占用名额，{date}前不完成内部激活会产生{amount}元违约费用并影响后续求职或学籍。",
        "evidence_type": "false_consequence",
        "normal_context": "回访您在{channel}主动投递或咨询的{topic}事项{ref}，只说明公开申请流程。",
        "normal_status": "不会以入职、助学或课程名额为由先收费，合同和费用必须在正式页面清楚展示。",
        "ambiguous": "来电自称招聘或课程顾问，只询问是否愿意了解公开岗位/课程，尚未断言录取、欠费或名额状态。",
        "ops": [("transfer", "先转{amount}元激活费到项目负责人的收款账户。"), ("unofficial_link", "点击群里发送的入职登记链接填写银行卡和身份信息。"), ("identity_document", "把手持身份证和学生证照片发到顾问私聊完成审核。")],
    },
}

PLATFORM_SCENARIOS = ["platform_membership", "million_guarantee", "automatic_debit", "trial_expiry"]
OTHER_SCENARIOS = [key for key in SCENARIOS if key not in PLATFORM_SCENARIOS]

PRE_CALLER = [
    "我先说明部门和事项范围，今天只围绕{topic}记录{ref}沟通。",
    "系统摘要只显示{date}和{city}的同步节点，具体页面暂时不在我这里。",
    "这条任务关联尾号{tail}，完整账户资料不需要您现在提供。",
    "我会先把时间、业务名称和记录来源分开讲，避免与其他通知混淆。",
    "当前只能看到服务侧摘要，尚未进入处理步骤。",
    "记录来源标为{device}，但这并不等于已经确认操作人。",
    "内部编号是{ref}，您可以先记下，不需要因为编号立即相信来电。",
    "这次联系发生在{date}{time}，地点标签为{city}，我先说明背景。",
    "我负责状态说明，稍后再解释为什么系统把这项任务分到人工席。",
    "前一条备注与当前号码尾号{tail}存在差异，我先把差异列出来。",
    "业务名称是{topic}，目前先核对是否与您近期接触的事项属于同一类。",
    "我不先要求您选择，只把系统里可见的几个字段按顺序念清楚。",
    "记录更新时间晚于创建时间，两个时间点需要分开理解。",
    "当前任务可能与家庭成员、员工或代理人使用{device}有关，但还需要核对。",
    "这通电话先停留在事实说明，后续是否处理要看记录状态。",
    "系统把任务归到{identity}名下，我先解释该岗位能看到哪些信息。",
    "我再重复一次参考号{ref}，后面每个状态都以这条记录为范围。",
    "来电显示不等于身份依据，我先把可独立核对的事实说完。",
    "如果您手边没有页面，可以只听日期和事项名称，不用同步操作。",
    "我注意到历史联系人可能已经变化，所以今天先不核完整身份资料。",
    "这条记录没有显示最终选择，只保留了一个待说明标记。",
    "我先确认您能否听清，背景声音不会改变事项本身。",
    "涉及金额时我会单独说明，现在先核对业务对象和时间。",
    "如果前后信息不一致，您可以随时打断，我只解释当前摘要。",
    "这项回访与其他订单或账户业务分开，不会合并处理。",
    "我把{ref}的生成时间和最近同步时间分别说明，免得混成一个日期。",
    "当前页只显示标签和状态，不包含支付明细或完整协议。",
]

PRE_CALLEE = [
    "我会把身份、日期和事项分开记录，不会因为你报了编号就操作。",
    "这个号码我不熟，你先讲能从公开渠道核验的事实。",
    "我手边没有完整页面，只能先听，后面会自己查询。",
    "家人或同事可能用过相关设备，所以我不会直接承认这条记录。",
    "时间和我记得的不完全一致，你先解释两个日期的含义。",
    "我现在在{city}，环境不方便操作手机，只能记录。",
    "尾号和事项我写下了，但身份还没有核实。",
    "我不会在电话里提供身份证、银行卡或验证码，你继续说事实。",
    "如果只是一般说明可以继续，涉及账户变更我会另外核验。",
    "我最近收到多条通知，不能把它们都合并成同一件事。",
    "你说的部门职责我不熟，需要和官网公开说明对照。",
    "我先确认这是不是旧记录回流，不会把提醒当成已办理。",
    "现在只算听取说明，不代表我同意任何选择。",
    "参考号我记了，接下来请说明记录为什么会关联到我。",
    "我会让旁边的人一起听，避免漏掉关键时间和金额。",
    "账号可能由多人使用，你不能只凭一个设备标签认定是我操作。",
    "我在工作间隙接听，请一句一句讲，不要跳到处理步骤。",
    "内部摘要不是最终依据，我最终只认自己查到的正式页面。",
    "你可以继续，但每出现一个新后果我都会重新判断。",
    "这一点我先保留，等你把来源和当前状态讲完整。",
    "我不急着下结论，先把事项名称、时间和主体对齐。",
    "请不要把我的沉默当成同意，我只是做记录。",
    "如果公开客服查不到，我不会因为电话解释就继续。",
    "我会按{ref}单独核对，不与其他业务混在一起。",
    "涉及{amount}元的说法必须有正式凭据，口头金额不能作为依据。",
    "你先说明当前是提醒、待确认还是已经发生的结果。",
    "我没法确认来电身份，后面任何建议都只作为待核信息。",
]

EARLY_CALLER = [
    "这个状态在内部页被标为需要人工处理，普通入口可能只显示概括信息。",
    "系统认为时间节点临近，所以才把{ref}转给人工席继续说明。",
    "如果您记忆和记录不一致，后台仍会按当前标签保留任务。",
    "这条状态不会因为挂断立即消失，后续仍会按既有规则推进。",
    "公开页面更新可能较慢，内部摘要暂时被作为处理依据。",
    "记录里把{date}设为关键节点，超过后会进入下一处理阶段。",
    "我只能按系统当前状态解释，无法用您口头否认直接删除记录。",
    "这项任务与尾号{tail}的历史联系方式绑定，所以更换设备也不会自动解除。",
    "后台要求先确认用户意愿，再决定是否释放该状态。",
    "如果今天不形成结果，系统会继续保留提醒并重复触发回访。",
    "内部流程把用户端页面和处理端状态分开，两边可能暂时不同步。",
    "您要求独立核验我可以记录，但当前节点仍按原规则计时。",
    "这条记录已经从普通咨询转到异常处理队列，状态不会自动回退。",
    "工作人员只能解释和登记，无法在电话中证明整个后台链路。",
    "系统把异议记在{ref}下，但异议本身不等于状态已经解除。",
    "我先把可能后果说完整，您再决定是否继续听处理方式。",
    "当前摘要没有显示实际处理人，只显示业务标签和节点。",
    "如果由家人或员工触发，后台仍会先联系账户登记人。",
    "任务备注要求在本次回访中说明状态，而不是等待普通通知。",
    "您可以质疑身份，但系统侧的倒计时不会因此暂停。",
    "我再强调一次，用户端没有看到并不能自动否定内部记录。",
    "这个节点只针对{topic}，不会和其他业务自动抵消。",
    "我会把您拒绝电话办理写入备注，但后果仍由系统规则决定。",
    "记录已经挂起一段时间，人工席被要求在{date}前完成说明。",
    "当前没有新的书面材料，处理依据仍是编号{ref}对应的状态。",
    "如果您稍后查询不到，可以再联系，但现在先把流程听完整。",
    "我理解您谨慎，不过后台把这项任务视为需要继续推进。",
]

EARLY_CALLEE = [
    "你已经在说具体后果了，我需要先核实这个状态是否真实存在。",
    "普通入口看不到不代表我要接受所谓内部流程。",
    "我不会因为倒计时或重复回访就跳过官方核验。",
    "如果系统和我的页面冲突，应由公开客服给出正式解释。",
    "你说的状态越来越具体，但没有提供可独立验证的依据。",
    "我只记录，不接受电话里的继续、关闭或处理选择。",
    "即使家人用过账号，也不能推出这项后果一定成立。",
    "我会保留异议，不会用口头确认替代正式页面。",
    "你把内部规则说清楚可以，但不能要求我先相信结果。",
    "涉及冻结、处罚、扣费或资格失效，我只认可书面依据。",
    "当前解释已经偏离普通客服流程，我准备挂断核验。",
    "我不会把未核实的后台标签当成自己的真实业务。",
    "如果这项记录确实存在，官方渠道应能留下同一编号。",
    "你不能一边说无法证明身份，一边要求我接受系统后果。",
    "我听到了，但不会在这通电话里形成处理结果。",
    "请把我的拒绝记录下来，我会自行联系对应机构。",
    "时间压力不能代替证据，我宁可错过所谓节点。",
    "我现在只确认你说过这些，不确认内容是真实的。",
    "后续如果出现敏感操作，我会立即停止通话。",
    "我会把{ref}带到公开渠道查询，不沿用你的处理路径。",
    "这个说法没有消除疑点，反而需要更严格核验。",
    "你先不要继续制造后果，我会用自己找到的号码联系。",
    "我不会因为你知道尾号或城市就认可身份。",
    "涉及{amount}元的后果需要正式账单，不是电话摘要。",
    "今天不会按你的规则选择，我先结束风险部分。",
    "你可以登记异议，但不要替我提交任何授权。",
    "我已经听明白，接下来只接受独立官方核验。",
]

EXPLICIT_CALLER = [
    "操作入口只在本次会话有效，退出后可能需要重新排队。",
    "系统要求保持通话在线，否则当前验证节点会自动失败。",
    "这一步只是验证，不会改变您其他账户的正常状态。",
    "如果页面出现风险提示，那是常规安全提醒，可以继续。",
    "不要切换到公开客服，两个通道同时处理会导致记录冲突。",
    "完成后我会给您回传结果编号，现在先保持页面不动。",
    "您可以先做最小步骤，后续内容等验证通过再解释。",
    "系统检测到暂停操作，所以需要您继续完成当前节点。",
    "这个流程由专席监督，不会在普通帮助页面留下完整说明。",
    "只要验证通过，前面提到的状态就会从后台释放。",
    "如果您担心，可以先处理一部分，确认结果后再继续。",
    "现在退出会被记为用户主动放弃处理，后果需要自行承担。",
    "我已经把您的异议备注了，但操作节点仍需本人完成。",
    "页面内容不要发给其他人，避免重复提交影响核验。",
    "请按我说的顺序做，不要自行返回或搜索其他入口。",
    "这项验证不会向您长期收费，只用于本次状态解除。",
    "系统还差最后一步，完成后就可以结束通话。",
    "如果出现失败提示，再按我后面说明的备用方式处理。",
    "当前记录已锁定到本次专席，换客服会从头开始。",
    "我会一直在线等您完成，不需要向身边的人解释。",
    "先不要挂断，验证结果只能在当前会话里同步。",
    "这是内部核验，不等同于普通支付或账户变更。",
    "操作完成后请告诉我页面最后一行显示的状态。",
    "如果设备有延迟，等待十秒再继续，不要退出页面。",
    "您的拒绝会被系统记录，但现在仍有一次补做机会。",
    "我再把步骤压缩说明，避免您因为页面切换而漏掉。",
    "当前通道即将关闭，请先完成已经开始的步骤。",
]

EXPLICIT_CALLEE = [
    "你已经提出具体操作，我不会继续。",
    "安全提醒不能被你解释成可以忽略，我现在停止。",
    "我不会保持通话操作账户，会先挂断核验。",
    "这一步涉及敏感权限，与你前面的身份说明必须分开判断。",
    "我拒绝按陌生来电的顺序操作，也不会回报页面状态。",
    "所谓专席通道无法替代公开流程，我不会进入。",
    "你要求我不要联系公开客服，这本身就不正常。",
    "我不会做最小金额或最小步骤测试。",
    "任何验证码、密码、转账或远程权限都不会提供。",
    "我会保留来电记录，后续由官方机构处理。",
    "你不用再解释操作，我已经决定挂断。",
    "即使前面的业务真实，也不代表这个路径安全。",
    "我不会把手机页面或账户信息展示给你。",
    "时间再紧也不会改变我的核验方式。",
    "如果确实有问题，正式渠道会留下可查通知。",
    "我不会让身边人回避，也不会对家人保密。",
    "你说验证不改变账户，但要求本身已经越界。",
    "我会从原 App、官网或柜台重新开始，不续接当前流程。",
    "这通电话到此为止，后面步骤不要再发。",
    "我不会报告任何成功或失败页面。",
    "不要再催，我不接受当前会话绑定的说法。",
    "我已经明确拒绝，继续施压不会让我操作。",
    "所有后果等我独立核实后再处理。",
    "我会把编号和话术一起提交给公开客服。",
    "这不是正常回访能要求的内容，我停止通话。",
    "你不需要等我，我不会完成所谓最后一步。",
    "我不会点击、下载、转账或提供任何动态信息。",
]

NORMAL_CALLER = [
    "本次只核对您主动发起的事项{ref}，不会据此新增其他业务。",
    "记录显示提交时间为{date}，您可以在{channel}看到同一时间。",
    "客服只能读取有限摘要，不需要您口述完整证件或账户资料。",
    "若当前不方便，可以挂断后自行进入原页面，工单不会因此失效。",
    "页面与我说法不一致时，以您主动打开的正式记录为准。",
    "付款、退款或费用必须由官方页面展示明确主体和金额。",
    "电话不会索取验证码、密码，也不会要求共享屏幕或远程控制。",
    "我先区分提醒、申请、办理和完成四个状态，避免误解。",
    "这条业务由您此前主动提交，系统没有把沉默视为授权。",
    "如果家人或同事协助过，仍应由您从正式页面核对结果。",
    "当前没有处罚、冻结或强制扣费，是否继续由您自行决定。",
    "您可以从官网公开电话重新呼入，工作人员能查询编号{ref}。",
    "涉及地址或设备变化，只需您在登录后的安全页面自己修改。",
    "我不会发送临时安装包、私人收款码或外部短链接。",
    "本次回访只说明{topic}，不会连带处理其他账户和订单。",
    "时间要求以正式通知为准，来电时间不是最后期限。",
    "如果页面没有对应项目，不需要为了配合电话补做操作。",
    "您可以让家人旁听，但不要把登录状态交给任何工作人员。",
    "金额{amount}元只是页面上的已知参考，最终以凭据为准。",
    "工作人员不能替您点击同意、关闭、续期或付款按钮。",
    "正式材料可在{channel}下载，电话摘要不能替代书面内容。",
    "如果决定取消，请由您从原申请入口提交，不必回拨本号码。",
    "工单{ref}只记录服务过程，不代表您已经确认任何收费。",
    "我们允许您先挂断核验身份，不会因核验而提高费用或失去资格。",
    "当前处理结果会同步到站内消息，您不用把截图发给来电人员。",
    "如需线下办理，请从官网查询网点，不接受电话指定的私人地址。",
    "这次没有要求您立即表态，稍后处理不会自动改变现有状态。",
]

NORMAL_CALLEE = [
    "这项业务是我主动发起的，我会从原页面核对编号。",
    "我接受你说明规则，但不会在来电中操作账户。",
    "页面时间能对上，后续我仍会自己查看完整记录。",
    "允许挂断重拨这一点合理，我会用自己找到的官方号码。",
    "验证码、密码和屏幕内容我不会提供。",
    "我只认可原支付路径和可验证的收款主体。",
    "提醒和实际扣费要分开，正式账单出来后我再判断。",
    "如果页面没有对应项目，我不会继续补操作。",
    "家人可以帮我核对，但账户权限仍由我本人掌握。",
    "我会把工单{ref}和手里的凭据逐项对照。",
    "当前没有强制后果，我可以等方便时再处理。",
    "完整地址和证件我会在登录后的页面自己维护。",
    "我不需要回拨当前号码，会从{channel}重新进入。",
    "这次只谈{topic}，不要扩展到其他账户。",
    "费用信息我会看正式页面，不按口头金额付款。",
    "我明白来电只做说明，不等于我已经授权变更。",
    "有差异我会暂停，不会为了时限跳过核验。",
    "我会保留正式通知，不把电话摘要当成最终结果。",
    "可以继续说明，但涉及安装、转账或远程操作我会停止。",
    "我自己能操作官方 App，不需要工作人员代办。",
    "如果需要线下材料，我会从官网查询网点和清单。",
    "站内消息能查到同一事项，这一点可以独立验证。",
    "我不会把截图和个人页面发到私人聊天。",
    "稍后处理不改变状态，我就按自己的时间核对。",
    "我已经区分申请时间和后台更新时间，不会混在一起。",
    "你把公开流程讲清楚即可，最终决定由我自己作出。",
    "目前闭环清楚，没有要求我交出敏感信息。",
]

AMB_CALLER = [
    "我先说明一般业务范围，不要求您现在确认任何账户状态。",
    "当前名单只包含回访联系方式，没有具体订单或协议内容。",
    "我可以介绍规则，但无法从这页判断您是否办理过{topic}。",
    "完整记录尚未加载，所以现在不适合下结论。",
    "这通电话只询问是否愿意了解公开流程，您可以拒绝。",
    "我没有看到扣费、处罚、冻结或待关闭状态。",
    "若您不认识该事项，可以挂断后自己从{channel}查询。",
    "目前还没有要求提交资料或执行操作。",
    "部门名称和工号可以先记下，但它们本身不能证明来电身份。",
    "这条任务可能只是批量调查，不等于用户存在具体业务。",
    "我先停留在主题说明，不把回访名单解释成个人登记。",
    "页面只写了{topic}分类，没有生效日期和处理结论。",
    "如果后续查到正式工单，再由公开客服继续说明。",
    "当前没有足够字段区分旧咨询、误留号码或真实业务。",
    "我不会因为您接听就记录同意或生成新的申请。",
    "参考号{ref}只是回访任务编号，不是业务订单号。",
    "如果信号中断，本次说明可以直接结束，不会产生后果。",
    "这项内容现在仍是待核信息，没有进入办理阶段。",
    "我只能确认来电主题，不能确认主体、金额或有效期。",
    "您可以要求结束，当前没有必须完成的步骤。",
]

AMB_CALLEE = [
    "目前只有身份和主题，我还不能判断这是不是我的业务。",
    "没有具体状态就先停在待核，不需要强行下结论。",
    "我会从公开入口查询，不依据回访名单确认。",
    "你没有提出异常操作，我先记录部门和事项。",
    "业务可能正常，也可能号码来源有误，现有信息不够。",
    "我不会提供敏感资料，但可以继续听一般规则。",
    "如果没有订单号和日期，我无法和自己的记录对应。",
    "当前没出现扣费或处罚后果，我不会提前判断。",
    "我先保留意见，等正式页面有记录再处理。",
    "回访任务和实际业务要分开，不能互相替代。",
    "你可以说明范围，但不要把它说成我已经办理。",
    "我没法核实身份，也没有看到明显异常流程。",
    "这段信息仍然两种解释都说得通。",
    "我会结束通话，之后自己查是否存在对应事项。",
    "当前只算一般询问，不代表我接受任何服务。",
    "如果后续出现具体要求，我会重新判断风险。",
    "参考号我记下了，但它不能证明个人业务状态。",
    "我先不把陌生身份当真，也不直接认定有问题。",
    "现有内容不足，我不会进入操作或付款环节。",
    "就到这里，我会用独立渠道确认主题是否真实。",
]


def stable_int(value: str, salt: str = "") -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:16], 16)


def allocate(total: int, keys: list[str], salt: str) -> dict[str, int]:
    base, rem = divmod(total, len(keys))
    result = {key: base for key in keys}
    order = sorted(keys, key=lambda key: stable_int(key, salt))
    for key in order[:rem]:
        result[key] += 1
    return result


def scenario_quota(split: str, label: str, total: int) -> dict[str, int]:
    if label != "fraud":
        return allocate(total, list(SCENARIOS), f"{split}-{label}")
    platform_total = total // 2
    result = allocate(platform_total, PLATFORM_SCENARIOS, f"{split}-platform")
    result.update(allocate(total - platform_total, OTHER_SCENARIOS, f"{split}-other"))
    return result


def bucket_values(bucket: str) -> list[int]:
    return {
        "2-8": [6, 8],
        "9-20": [10, 12, 14, 16, 18, 20],
        "21-40": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
        "41+": [42, 44, 46, 48, 50, 52, 54],
    }[bucket]


def make_context(index: int, scenario_key: str, split: str) -> dict[str, object]:
    spec = SCENARIOS[scenario_key]
    key = f"{index}|{scenario_key}|{split}"
    rng = random.Random(stable_int(key, "context"))
    ref = f"{chr(65 + index % 26)}{(stable_int(key, 'ref') % 900000) + 100000}"
    return {
        "organization": rng.choice(spec["organizations"]),
        "identity": rng.choice(spec["identities"]),
        "topic": rng.choice(spec["topics"]),
        "city": rng.choice(CITIES),
        "time": rng.choice(TIMES),
        "date": rng.choice(MONTH_DAYS),
        "person": rng.choice(PEOPLE),
        "device": rng.choice(DEVICES),
        "channel": rng.choice(SCENARIO_CHANNELS.get(scenario_key, CHANNELS)),
        "amount": rng.choice(AMOUNTS) + index % 17,
        "tail": f"{(stable_int(key, 'tail') % 10000):04d}",
        "ref": ref,
    }


def format_text(template: str, ctx: dict[str, object]) -> str:
    return template.format(**ctx)


def choose_from(items: list[str], call_index: int, turn_id: int, salt: str, ctx: dict[str, object]) -> str:
    offset = stable_int(f"{call_index}|{salt}") % len(items)
    choice = items[(offset + (turn_id // 2) * 7) % len(items)]
    return format_text(choice, ctx)


def nearest_caller_turn(n_turns: int, bin_name: str, key: str, needs_later: bool) -> int:
    bounds = {"Q1": (0.0, 0.25), "Q2": (0.25, 0.50), "Q3": (0.50, 0.75), "Q4": (0.75, 1.0)}[bin_name]
    candidates = []
    for turn_id in range(1, n_turns + 1, 2):
        ratio = turn_id / n_turns
        if bounds[0] < ratio <= bounds[1] and (not needs_later or turn_id + 2 <= n_turns):
            candidates.append(turn_id)
    if not candidates:
        candidates = [turn_id for turn_id in range(1, n_turns + 1, 2) if not needs_later or turn_id + 2 <= n_turns]
    return candidates[stable_int(key, "te") % len(candidates)]


def stage_trajectory(decision: str, n_turns: int, t_s: int | None, t_e: int | None, t_a: int | None) -> list[dict]:
    result = []
    for turn_id in range(1, n_turns + 1):
        if decision == "normal":
            current = "normal"
        elif decision == "ambiguous":
            current = "normal" if turn_id < t_s else "ambiguous"
        elif turn_id < t_s:
            current = "normal"
        elif turn_id < t_e:
            current = "ambiguous"
        elif t_a is not None and turn_id >= t_a:
            current = "explicit_fraud"
        else:
            current = "early_fraud"
        result.append({"prefix_end_turn": turn_id, "decision": current, "stage_target": STAGE_TARGET[current]})
    return result


def build_dialogue(item: dict, global_index: int) -> tuple[list[dict], dict]:
    scenario_key = item["scenario"]
    spec = SCENARIOS[scenario_key]
    ctx = make_context(global_index, scenario_key, item["split"])
    n_turns = item["n_turns"]
    category = item["category"]
    decision = item["decision"]
    t_s = 1 if category != "normal" else None
    t_e = item.get("tE")
    t_a = item.get("tA")
    operation_type = None
    operation_text = None
    if decision == "explicit_fraud":
        op = spec["ops"][stable_int(f"{global_index}|{scenario_key}", "op") % len(spec["ops"])]
        operation_type, operation_template = op
        operation_text = format_text(operation_template, ctx)

    turns = []
    for turn_id in range(1, n_turns + 1):
        caller = turn_id % 2 == 1
        if category == "fraud":
            if caller:
                if turn_id == 1:
                    if t_e == 1:
                        text = f"您好，我是{ctx['organization']}的{ctx['identity']}。" + format_text(spec["fraud_evidence"], ctx)
                    else:
                        text = f"您好，我是{ctx['organization']}的{ctx['identity']}，联系您说明{ctx['topic']}的一项记录。"
                    text += f"本次参考号{ctx['ref']}，摘要时间为{ctx['date']}{ctx['time']}，地点备注{ctx['city']}。"
                elif turn_id == t_e:
                    text = format_text(spec["fraud_evidence"], ctx)
                elif turn_id == t_a:
                    text = operation_text
                elif t_e is not None and turn_id > t_e and (t_a is None or turn_id < t_a):
                    text = choose_from(EARLY_CALLER, global_index, turn_id, "early-caller", ctx)
                elif t_a is not None and turn_id > t_a:
                    text = choose_from(EXPLICIT_CALLER, global_index, turn_id, "explicit-caller", ctx)
                else:
                    text = choose_from(PRE_CALLER, global_index, turn_id, "pre-caller", ctx)
            else:
                previous = turns[-1]["turn_id"]
                if t_a is not None and previous >= t_a:
                    text = choose_from(EXPLICIT_CALLEE, global_index, turn_id, "explicit-callee", ctx)
                elif t_e is not None and previous >= t_e:
                    text = choose_from(EARLY_CALLEE, global_index, turn_id, "early-callee", ctx)
                else:
                    text = choose_from(PRE_CALLEE, global_index, turn_id, "pre-callee", ctx)
        elif category == "normal":
            if caller:
                if turn_id == 1:
                    text = f"您好，这里是{ctx['organization']}的{ctx['identity']}。" + format_text(spec["normal_context"], ctx)
                    text += f"摘要还记录了{ctx['date']}{ctx['time']}、{ctx['city']}和设备{ctx['device']}，这些仅供您独立核对。"
                elif turn_id == 3:
                    text = format_text(spec["normal_status"], ctx)
                else:
                    text = choose_from(NORMAL_CALLER, global_index, turn_id, "normal-caller", ctx)
            else:
                text = choose_from(NORMAL_CALLEE, global_index, turn_id, "normal-callee", ctx)
        else:
            if caller:
                if turn_id == 1:
                    text = f"您好，我是{ctx['organization']}的{ctx['identity']}。" + format_text(spec["ambiguous"], ctx)
                    text += f"当前只有回访号{ctx['ref']}和{ctx['date']}的摘要，没有个人业务结论。"
                else:
                    text = choose_from(AMB_CALLER, global_index, turn_id, "amb-caller", ctx)
            else:
                text = choose_from(AMB_CALLEE, global_index, turn_id, "amb-callee", ctx)
        turns.append({"turn_id": turn_id, "speaker": "来电人" if caller else "接电人", "text": text})

    metadata = {
        "ctx": ctx,
        "tS": t_s,
        "tE": t_e,
        "tA": t_a,
        "operation_type": operation_type,
        "operation_text": operation_text,
    }
    return turns, metadata


def render_text(turns: list[dict]) -> str:
    return "\n".join(f"{turn['turn_id']:03d} {turn['speaker']}：{turn['text']}" for turn in turns)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def make_plan() -> list[dict]:
    plan = []
    serial = 0
    for split, quotas in SPLIT_QUOTAS.items():
        for category, total in quotas.items():
            for scenario, count in scenario_quota(split, category, total).items():
                for local_index in range(count):
                    serial += 1
                    plan.append({"serial": serial, "split": split, "category": category, "scenario": scenario, "local_index": local_index})
    if len(plan) != 2000:
        raise ValueError(f"plan rows={len(plan)}")

    for category, quotas in CLASS_LENGTH_QUOTAS.items():
        indices = [index for index, item in enumerate(plan) if item["category"] == category]
        rng = random.Random(stable_int(category, "bucket-shuffle"))
        rng.shuffle(indices)
        buckets = [bucket for bucket, count in quotas.items() for _ in range(count)]
        rng.shuffle(buckets)
        for index, bucket in zip(indices, buckets):
            item = plan[index]
            values = bucket_values(bucket)
            item["length_bucket"] = bucket
            item["n_turns"] = values[stable_int(f"{item['serial']}|{bucket}", "nturn") % len(values)]

    fraud_indices = [index for index, item in enumerate(plan) if item["category"] == "fraud"]
    rng = random.Random(stable_int("fraud", "fraud-shuffle"))
    rng.shuffle(fraud_indices)
    bins = [bin_name for bin_name, count in TE_BIN_QUOTAS.items() for _ in range(count)]
    rng.shuffle(bins)
    nonshort = [index for index in fraud_indices if plan[index]["length_bucket"] != "2-8"]
    explicit_set = set(nonshort[:600])
    for index, bin_name in zip(fraud_indices, bins):
        item = plan[index]
        explicit = index in explicit_set
        item["decision"] = "explicit_fraud" if explicit else "early_fraud"
        item["evidence_position_bin"] = bin_name
        # Six-turn calls have no odd caller turn in Q3: turn 3 is exactly Q2
        # and turn 5 is already Q4. Keep the sample in the 2-8 bucket while
        # making the requested evidence-position bin attainable.
        if bin_name == "Q3" and item["n_turns"] == 6:
            item["n_turns"] = 8
        if explicit and bin_name == "Q4" and item["n_turns"] < 14:
            item["n_turns"] = 14
        t_e = nearest_caller_turn(item["n_turns"], bin_name, str(item["serial"]), explicit)
        item["tE"] = t_e
        item["tA"] = t_e + 2 if explicit else None

    for item in plan:
        if item["category"] == "normal":
            item["decision"] = "normal"
            item["evidence_position_bin"] = None
        elif item["category"] == "ambiguous":
            item["decision"] = "ambiguous"
            item["evidence_position_bin"] = None
    return plan


def build_record(item: dict, index: int) -> dict:
    turns, meta = build_dialogue(item, index)
    full_text = render_text(turns)
    decision = item["decision"]
    category = item["category"]
    final_binary = 1 if category == "fraud" else 0 if category == "normal" else -1
    call_id = f"V365_{item['split'].upper()}_{index:04d}"
    sample_id = f"V365_FULL_{index:04d}"
    op_flags = {name: False for name in OPERATION_TYPES}
    operation_events = []
    if meta["operation_type"] is not None:
        op_flags[meta["operation_type"]] = True
        operation_events.append({"turn_id": meta["tA"], "quote": turns[meta["tA"] - 1]["text"], "operation_type": meta["operation_type"]})
    core_evidence = []
    supporting_evidence = []
    if category == "fraud":
        core_evidence = [{
            "turn_id": meta["tE"],
            "quote": turns[meta["tE"] - 1]["text"],
            "evidence_type": SCENARIOS[item["scenario"]]["evidence_type"],
            "explanation": "甲方严格边界下，当前来电已形成身份主张、个体化状态断言与异常生命周期、处理或后果链条。",
        }]
        supporting_evidence = [{
            "turn_id": 1,
            "quote": turns[0]["text"],
            "evidence_type": "identity_anomaly",
            "explanation": "来电人以平台、权威机构、金融服务、熟人关系或业务机构身份推进个体化事项。",
        }]
    elif category == "ambiguous":
        supporting_evidence = [{
            "turn_id": 1,
            "quote": turns[0]["text"],
            "evidence_type": "identity_anomaly",
            "explanation": "陌生身份与敏感主题需要继续观察，但当前尚无个体化异常状态和处理链。",
        }]

    blueprint_family = f"{item['split']}|{item['scenario']}|{category}|{item['length_bucket']}|route{stable_int(str(index), 'route') % 17:02d}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "policy_version": POLICY_VERSION,
        "call_id": call_id,
        "sample_id": sample_id,
        "source_sample_id": f"synthetic:{DATASET_VERSION}:{index:04d}",
        "content_group_id": f"V365_GROUP_{digest(blueprint_family)[:16]}",
        "dialogue_blueprint_id": blueprint_family,
        "duplicate_cluster_id": call_id,
        "split": item["split"],
        "source_type": "synthetic_compositional",
        "source_name": "v365_2000_strict_boundary_generator",
        "source_record_sha256": digest(json.dumps(turns, ensure_ascii=False, sort_keys=True)),
        "text_sha256": digest(full_text),
        "turns": turns,
        "n_full_turns": len(turns),
        "prefix_end_turn": len(turns),
        "n_prefix_turns": len(turns),
        "prefix_text": full_text,
        "model_input_text": full_text,
        "prefix_text_hash": digest(full_text),
        "normalized_text_hash": digest(normalize(full_text)),
        "language": "zh-CN",
        "scenario_tag": item["scenario"],
        "claimed_identity_type": meta["ctx"]["identity"],
        "length_bucket": item["length_bucket"],
        "evidence_position_bin": item["evidence_position_bin"],
        "final_label": category,
        "final_binary_label": final_binary,
        "decision": decision,
        "stage_target": STAGE_TARGET[decision],
        "binary_target": final_binary,
        "binary_loss_mask": 0 if category == "ambiguous" else 1,
        "stage_loss_mask": 0,
        "sample_weight": 1.0,
        "prefix_relation": "clean_normal" if category == "normal" else None,
        "tS_turn": meta["tS"],
        "tE_turn": meta["tE"],
        "tA_turn": meta["tA"],
        "tFinal_turn": len(turns),
        "early_detectable": category == "fraud",
        "explicit_operation_present": op_flags,
        "operation_events": operation_events,
        "core_evidence": core_evidence,
        "supporting_evidence": supporting_evidence,
        "insufficient_reason": "当前只有身份、主题或一般回访，尚未形成个体化异常状态与处理/后果链。" if category == "ambiguous" else "",
        "boundary_rationale": (
            "身份主张、个体化状态与异常后果链已完整形成，不等待明确操作即为 early_fraud。"
            if category == "fraud" else
            "用户主动事项、正式系统可核验、无异常后果与敏感操作，构成困难正常闭环。"
            if category == "normal" else
            "当前信息不足，正常与异常解释仍可同时成立，保留 ambiguous。"
        ),
        "turn_stage_labels": stage_trajectory(decision, len(turns), meta["tS"], meta["tE"], meta["tA"]),
        "annotation_method": "deterministic_compositional_generation_with_rule_labels",
        "annotator_id": "v365_generator",
        "annotation_model": None,
        "annotation_prompt_version": POLICY_VERSION,
        "reviewer_id": "deterministic_validator_pending_independent_reviewer",
        "adjudicator_id": None,
        "confidence": "medium" if category == "ambiguous" else "high",
        "review_required": True,
        "review_status": "pending_independent_review",
        "disagreement_type": None,
        "future_content_used": False,
        "label_leakage_flag": False,
        "quality_flags": ["synthetic_source_ratio_gap", "independent_review_pending"],
        "annotation_timestamp": TIMESTAMP,
    }
    return record


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def blind_input(record: dict) -> dict:
    # Use a positive allowlist.  The previous negative list accidentally left
    # evidence_position_bin and review metadata in the blind artifact; the
    # former perfectly separated fraud from non-fraud records.
    allowed = [
        "schema_version", "dataset_version", "policy_version", "call_id", "sample_id",
        "split", "turns", "n_full_turns", "prefix_end_turn", "n_prefix_turns",
        "prefix_text", "model_input_text", "text_sha256", "prefix_text_hash",
        "normalized_text_hash", "language", "scenario_tag", "claimed_identity_type",
        "length_bucket", "tFinal_turn",
    ]
    return {key: record[key] for key in allowed}


def blind_label(record: dict) -> dict:
    keys = [
        "call_id", "sample_id", "final_label", "final_binary_label", "decision", "stage_target", "binary_target",
        "binary_loss_mask", "stage_loss_mask", "sample_weight", "prefix_relation", "tS_turn", "tE_turn", "tA_turn",
        "tFinal_turn", "early_detectable", "explicit_operation_present", "operation_events", "core_evidence",
        "supporting_evidence", "insufficient_reason", "boundary_rationale", "turn_stage_labels", "confidence",
        "review_required", "review_status", "quality_flags",
    ]
    return {key: record[key] for key in keys}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    plan = make_plan()
    records = [build_record(item, index) for index, item in enumerate(plan, 1)]
    by_split = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)

    write_jsonl(DATA / "v365_calls_train.jsonl", by_split["train"])
    write_jsonl(DATA / "v365_calls_dev.jsonl", by_split["dev"])
    write_jsonl(DATA / "v365_calls_stress.jsonl", by_split["stress"])
    write_jsonl(DATA / "v365_calls_blind_input.jsonl", [blind_input(record) for record in by_split["blind"]])
    write_jsonl(DATA / "restricted_v365_calls_blind_labels.jsonl", [blind_label(record) for record in by_split["blind"]])

    stats = {
        "schema_version": "v365.statistics.v1",
        "dataset_version": DATASET_VERSION,
        "calls": len(records),
        "split_counts": {split: len(rows) for split, rows in sorted(by_split.items())},
        "split_label_counts": {
            split: dict(sorted(Counter(row["final_label"] for row in rows).items()))
            for split, rows in sorted(by_split.items())
        },
        "decision_counts": dict(sorted(Counter(row["decision"] for row in records).items())),
        "final_label_counts": dict(sorted(Counter(row["final_label"] for row in records).items())),
        "scenario_counts": dict(sorted(Counter(row["scenario_tag"] for row in records).items())),
        "fraud_scenario_counts": dict(sorted(Counter(row["scenario_tag"] for row in records if row["final_label"] == "fraud").items())),
        "length_bucket_counts": dict(sorted(Counter(row["length_bucket"] for row in records).items())),
        "fraud_evidence_position_counts": dict(sorted(Counter(row["evidence_position_bin"] for row in records if row["final_label"] == "fraud").items())),
        "early_detectable_before_operation": sum(row["final_label"] == "fraud" and (row["tA_turn"] is None or row["tE_turn"] < row["tA_turn"]) for row in records),
        "explicit_operation_calls": sum(row["decision"] == "explicit_fraud" for row in records),
        "difficult_normal_total": sum(row["final_label"] == "normal" for row in records),
        "difficult_normal_sensitive_lexicon_count": sum(
            row["final_label"] == "normal"
            and bool(re.search(r"验证码|密码|转账|付款|银行卡|屏幕|远程|链接|收款|冻结|处罚|扣费|数字资产|SIM|eSIM", row["prefix_text"], re.I))
            for row in records
        ),
        "source_type_counts": dict(sorted(Counter(row["source_type"] for row in records).items())),
        "unique_call_ids": len({row["call_id"] for row in records}),
        "unique_text_sha256": len({row["text_sha256"] for row in records}),
        "unique_normalized_text_sha256": len({row["normalized_text_hash"] for row in records}),
        "review_status_counts": dict(sorted(Counter(row["review_status"] for row in records).items())),
        "spec_conflicts_and_release_gaps": [
            "第八节 MVP 规定诈骗共 800 通，但 8.2 同时要求至少 1000 通在明确操作前可标 early_fraud；精确 2000 通方案无法同时满足。本批 800/800 通诈骗全部可在操作前识别。",
            "第八节 MVP 规定诈骗共 800 通，但 8.2 要求官方 12 场景每类不少于 80 通（至少 960）且平台类合计不少于 400；精确 2000 通的 800 诈骗无法同时满足。本批平台四类合计 400，其余场景记录实际缺口。",
            "全部为 synthetic_compositional，未达到公开真实、脱敏真实或人工自然改写至少 25% 的来源比例。",
            "未完成独立双标；所有记录 review_required=true、review_status=pending_independent_review，stage_loss_mask=0，不应直接进入阶段监督训练。",
            "本批使用同一确定性生成程序；虽然全文、规范化哈希唯一且执行近重复检查，仍需人工排查组合式模板痕迹。",
        ],
    }
    (DATA / "v365_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
