from __future__ import annotations

from text_model.data.wordpiece_counter import BertWordPieceCounter
from text_model.scripts.data.finalize_longdoc_subagent_results import (
    DEFAULT_VOCAB,
    MAX_REPEATED_CLAUSE_RATIO,
    caller_asserts_risk_action,
    normal_caller_advances_unsafe_action,
    render_turns,
    select_slots,
    validate_result,
)


def _task(kind="fraud"):
    return {
        "task_id": "T1",
        "slot_id": "S1",
        "slot_index": 0,
        "split": "train",
        "candidate_variant": 0,
        "generation_kind": kind,
        "content_group": "target_fraud" if kind == "fraud" else "target_normal",
        "scenario_tag": "douyin_live_membership",
        "label_13": 1 if kind == "fraud" else 0,
        "subcategory_code": "01_0101" if kind == "fraud" else "00_0000",
        "target_wordpiece_min": 1,
        "target_wordpiece_max": 500,
        "length_bucket": "wp_385_512",
        "evidence_position_bin": "Q1" if kind == "fraud" else "normal",
        "cross_window_evidence": False,
        "caller_number_type": "unknown",
    }


def _fraud_result():
    texts = [
        ("caller", "这里是抖音会员客服，请您现在下载指定软件并开启屏幕共享。"),
        ("callee", "请先说明具体业务和官方核验方式。"),
        ("caller", "您的直播会员显示即将到期，需要今天处理。"),
        ("callee", "我会先从抖音官方应用查看会员状态。"),
        ("caller", "处理期间请保持电话畅通，我会说明后续步骤。"),
        ("callee", "涉及账户操作时我需要独立确认。"),
        ("caller", "系统页面会显示对应项目，您按页面提示继续。"),
        ("callee", "我拒绝远程操作，并会联系官方客服。"),
    ]
    return {
        "task_id": "T1",
        "turns": [{"speaker": speaker, "text": text} for speaker, text in texts],
        "context_evidence": [],
        "risk_evidence": ["请您现在下载指定软件并开启屏幕共享"],
        "scenario_summary": "冒充抖音会员客服诱导远控",
        "generation_notes": "测试",
    }


def test_generated_roles_are_preserved_but_excluded_from_model_text():
    counter = BertWordPieceCounter(DEFAULT_VOCAB)
    row = validate_result(_task(), _fraud_result(), counter)
    assert row["automatic_validation"] == "READY"
    assert "来电人：" in row["text"]
    assert "接电人：" in row["audio_text"]
    assert "来电人：" not in row["model_input_text"]
    assert row["evidence_token_ranges"]
    assert row["chunk_token_ranges"]
    assert row["role_marker_token_policy"] == "excluded_from_model_input_and_wordpiece_counts"


def test_evidence_role_prefix_is_normalized_at_input_boundary():
    result = _fraud_result()
    result["risk_evidence"] = ["来电人：请您现在下载指定软件并开启屏幕共享"]
    row = validate_result(_task(), result, BertWordPieceCounter(DEFAULT_VOCAB))
    assert row["automatic_validation"] == "READY"
    assert row["risk_evidence"] == ["请您现在下载指定软件并开启屏幕共享"]


def test_adjacent_speaker_segments_are_merged_at_input_boundary():
    raw_turns = [
        {"speaker": "caller", "text": "第一段。"},
        {"speaker": "caller", "text": "第二段。"},
        {"speaker": "callee", "text": "第三段。"},
        {"speaker": "caller", "text": "第四段。"},
        {"speaker": "callee", "text": "第五段。"},
        {"speaker": "caller", "text": "第六段。"},
        {"speaker": "callee", "text": "第七段。"},
        {"speaker": "caller", "text": "第八段。"},
        {"speaker": "callee", "text": "第九段。"},
    ]
    turns, marked_text, _, errors = render_turns(raw_turns)
    assert errors == []
    assert len(turns) == 8
    assert turns[0] == {"speaker": "caller", "text": "第一段。\n第二段。"}
    assert marked_text.count("来电人：") == 4


def test_slot_selection_requires_independent_acceptance_and_reports_missing_slots():
    counter = BertWordPieceCounter(DEFAULT_VOCAB)
    row = validate_result(_task(), _fraud_result(), counter)
    selected, unresolved = select_slots([row], {"T1": "accept"}, ["S1", "S2"])
    assert len(selected) == 1
    assert selected[0]["review_status"] == "accepted"
    assert unresolved == ["S2"]


def test_mechanical_near_duplicate_turns_are_rejected():
    result = {
        "task_id": "T1",
        "turns": [
            {"speaker": "caller", "text": "这里是抖音客服，请您下载指定软件，再继续听我说明第一项会员流程。"},
            {"speaker": "callee", "text": "我先确认一下来电目的，随后会从官方应用核验。"},
            {"speaker": "caller", "text": "这里是抖音客服，请您核对会员页面，再继续听我说明第二项会员流程。"},
            {"speaker": "callee", "text": "页面暂时没有异常，我需要知道服务的具体名称。"},
            {"speaker": "caller", "text": "这里是抖音客服，请您核对会员页面，再继续听我说明第三项会员流程。"},
            {"speaker": "callee", "text": "涉及账户操作时，我会挂机后联系正式客服。"},
            {"speaker": "caller", "text": "这里是抖音客服，请您核对会员页面，再继续听我说明第四项会员流程。"},
            {"speaker": "callee", "text": "我拒绝按陌生来电的指示继续操作。"},
        ],
        "context_evidence": [],
        "risk_evidence": ["请您下载指定软件"],
        "scenario_summary": "机械复读测试",
        "generation_notes": "测试",
    }
    row = validate_result(_task(), result, BertWordPieceCounter(DEFAULT_VOCAB))
    assert row["automatic_validation"] == "REJECT"
    assert "near_duplicate_turn_ratio_high" in row["automatic_validation_errors"]


def test_risk_detection_distinguishes_warnings_from_asserted_actions():
    assert not normal_caller_advances_unsafe_action(
        "正规客服不会要求您开启屏幕共享，也不会让费用转到个人账户。"
    )
    assert not normal_caller_advances_unsafe_action(
        "有人声称账户涉案并要求把钱转到安全账户，这不属于正常核查流程。"
    )
    assert not normal_caller_advances_unsafe_action(
        "您当时描述过对方让您安装陌生软件，但举报工单没有保存敏感信息。"
    )
    assert not normal_caller_advances_unsafe_action(
        "任何要求下载陌生软件、共享屏幕或提供验证码的做法都不属于回访流程。"
    )
    assert not normal_caller_advances_unsafe_action(
        "遇到让下载陌生会议软件、打开屏幕共享的要求，可以先终止通话。"
    )
    assert not normal_caller_advances_unsafe_action(
        "我们主要确认当时有没有转账、共享屏幕、提供验证码或安装陌生软件。"
    )
    assert normal_caller_advances_unsafe_action("现在把钱转到安全账户。")
    assert normal_caller_advances_unsafe_action("现在请安装陌生软件并开启屏幕共享。")
    assert caller_asserts_risk_action("你先转760元到我发给你的收款账号。")
    assert caller_asserts_risk_action("把手机收到的六位验证码直接读给我。")
    assert caller_asserts_risk_action("点开申报链接并把屏幕共享给我。")
    assert caller_asserts_risk_action(
        "平台今天排不上，你先把六百八十元预约金转到个人收款码。"
    )
    assert caller_asserts_risk_action("安装后收到六位校验码，直接念给我。")
    assert caller_asserts_risk_action("我急用8600元，你先转到我亲属的卡上。")
    assert caller_asserts_risk_action("您登录后把页面共享给我，我来确认地区字段。")
    assert caller_asserts_risk_action("先往我个人收款码付五百元订金。")
    assert caller_asserts_risk_action("您先在活动钱包充值一百元做体验。")
    assert caller_asserts_risk_action("验证码需要客服录入，不然后台无法确认。")
    assert caller_asserts_risk_action("你先转一千八给我补上行李费。")
    assert caller_asserts_risk_action("下一单需要你先垫付一千二百元下单。")
    assert caller_asserts_risk_action("能不能先帮我垫一千五百元办手续。")
    assert caller_asserts_risk_action("需要先下载保单协同助手，再报出八位核验码。")
    assert caller_asserts_risk_action(
        "如果页面显示验证金，就按我给的收款信息先转一笔。"
    )
    assert not caller_asserts_risk_action("平台不会要求你把钱转到个人账户。")
    assert MAX_REPEATED_CLAUSE_RATIO == 0.10


def test_cross_window_context_and_risk_are_audited_separately():
    task = _task()
    task.update(
        {
            "target_wordpiece_min": 500,
            "target_wordpiece_max": 1800,
            "length_bucket": "wp_513_1024",
            "evidence_position_bin": "Q4",
            "cross_window_evidence": True,
        }
    )
    fillers = [
        "我先说明完整业务背景和当前页面状态，您可以逐项询问，我会按流程解释相关信息。" * 2,
        "我需要先确认来电目的，涉及账户的事项会通过官方应用独立核验，请继续说明。" * 2,
        "系统记录显示您之前使用过抖音直播会员服务，目前需要核对会员权益和登记状态。" * 2,
        "我正在查看自己的应用页面，暂时没有看到异常提醒，也不会离开官方渠道操作。" * 2,
        "这项业务涉及多个处理步骤，我先把到期规则、服务周期和页面入口依次解释清楚。" * 2,
        "您继续讲，我会记录每一步，但任何敏感信息和资金动作都要先确认真实性。" * 2,
        "前面的身份登记已经说明完毕，下面进入账户处理阶段，请保持电话畅通并查看页面。" * 2,
        "我可以听取说明，不过仍会自行核验，不会因为催促就直接执行账户操作。" * 2,
        "请您现在下载指定软件并开启屏幕共享，再把收到的验证码告诉我完成会员关闭。" * 2,
        "我拒绝下载、共享屏幕和提供验证码，随后会从抖音官方应用联系正式客服核实。" * 2,
    ]
    result = {
        "task_id": "T1",
        "turns": [
            {
                "speaker": "caller" if index % 2 == 0 else "callee",
                "text": text,
            }
            for index, text in enumerate(fillers)
        ],
        "context_evidence": ["系统记录显示您之前使用过抖音直播会员服务"],
        "risk_evidence": [
            "请您现在下载指定软件并开启屏幕共享，再把收到的验证码告诉我"
        ],
        "scenario_summary": "抖音会员跨窗口诱导远控",
        "generation_notes": "跨窗口测试",
    }
    counter = BertWordPieceCounter(DEFAULT_VOCAB)
    row = validate_result(task, result, counter)
    assert row["automatic_validation"] == "READY", row["automatic_validation_errors"]
    assert row["context_evidence_token_ranges"]
    assert row["evidence_token_ranges"]
    assert len(row["chunk_token_ranges"]) >= 2
