"""
app.py — 专业基金持仓分析系统 · Streamlit 主程序
==================================================
定位：基金经理辅助决策工具，聚焦市场数据与风险分析
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_manager import (
    USE_MOCK_DATA,
    DEEPSEEK_API_KEY,
    WARNING_YELLOW,
    WARNING_RED,
    load_api_key,
    save_api_key,
    load_holdings,
    save_holdings,
    get_all_snapshots,
    generate_mock_index_data,
    fetch_index_data,
    analyze_industry_concentration,
    analyze_hedging,
    analyze_portfolio_health,
    generate_recommendations,
    generate_ai_briefing,
    get_random_term,
    GLOSSARY_TERMS,
)

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="📊 基金分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown("""<style>
    .main .block-container { padding-top: 1rem; }
    .index-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 20px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .index-card.up { border-bottom: 3px solid #ef5350; }
    .index-card.down { border-bottom: 3px solid #66bb6a; }
    .warning-yellow { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 8px; margin: 8px 0; color: #856404; }
    .warning-red { background: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; border-radius: 8px; margin: 8px 0; color: #721c24; }
    .briefing-section { background: #f8f9fa; border-radius: 12px; padding: 20px; margin: 10px 0; border: 1px solid #e0e0e0; }
    .briefing-section h4 { color: #333; margin-bottom: 8px; }
    .term-card { background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%); border-radius: 12px; padding: 16px; margin: 10px 0; border-left: 4px solid #3f51b5; }
</style>""", unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown("## 📊 基金分析系统")
    st.caption("专业基金经理辅助决策")

    st.markdown("---")
    st.subheader("⚙️ 数据设置")
    use_mock = st.toggle("📡 模拟数据", value=USE_MOCK_DATA)

    # API Key
    st.markdown("---")
    saved_key = DEEPSEEK_API_KEY or load_api_key()
    ai_key = st.text_input(
        "DeepSeek Key", value=saved_key, type="password",
        placeholder="sk-...",
        help="填入后AI简报自动保存",
        key="ai_key_input",
    )
    use_ai = bool(ai_key)
    if ai_key and ai_key != saved_key:
        save_api_key(ai_key)

    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ---- 持仓基金管理 ----
    st.subheader("📋 持仓基金")

    # 初始化 session state
    if "fund_holdings" not in st.session_state:
        st.session_state["fund_holdings"] = load_holdings()

    # 添加新基金
    with st.expander("➕ 添加基金", expanded=False):
        new_name = st.text_input("基金名称", placeholder="例：易方达蓝筹精选混合", key="new_fund_name")
        new_code = st.text_input("基金代码", placeholder="例：005827", key="new_fund_code")
        new_sector = st.text_input("所属赛道", placeholder="例：消费/蓝筹", key="new_fund_sector")
        if st.button("确认添加", use_container_width=True):
            name_ok = new_name.strip()
            code_ok = new_code.strip().isdigit() and len(new_code.strip()) == 6
            sector_ok = new_sector.strip()
            if not name_ok:
                st.error("请输入基金名称")
            elif not code_ok:
                st.error("请输入6位数字基金代码")
            elif not sector_ok:
                st.error("请输入所属赛道")
            elif any(f["code"] == new_code.strip() for f in st.session_state["fund_holdings"]):
                st.error("该基金代码已存在")
            else:
                st.session_state["fund_holdings"].append({
                    "name": new_name.strip(),
                    "code": new_code.strip(),
                    "sector": new_sector.strip(),
                })
                save_holdings(st.session_state["fund_holdings"])
                st.cache_data.clear()
                st.success(f"✅ 已添加 {new_name.strip()}")
                st.rerun()

    # 当前持仓列表（可删除）
    for i, fund in enumerate(st.session_state["fund_holdings"]):
        col_name, col_del = st.columns([4, 1])
        with col_name:
            st.caption(f"📌 {fund['name']} `{fund['code']}`")
        with col_del:
            if st.button("✕", key=f"del_{fund['code']}", help=f"删除 {fund['name']}"):
                st.session_state["fund_holdings"].pop(i)
                save_holdings(st.session_state["fund_holdings"])
                st.cache_data.clear()
                st.rerun()

    st.markdown("---")
    st.caption(f"更新时间: {datetime.now().strftime('%m-%d %H:%M')}")


# ============================================================
# 加载数据
# ============================================================

@st.cache_data(ttl=1800, show_spinner="🌐 获取市场数据...")
def load_data(use_mock: bool, holdings: tuple = None):
    # 将缓存友好的 tuple 格式转回 list[dict]
    # None = 未传，() = 空持仓，非空 tuple = 有持仓
    fund_list = None if holdings is None else [{"name": h[0], "code": h[1], "sector": h[2]} for h in holdings]
    snaps = get_all_snapshots(use_mock=use_mock, holdings=fund_list)
    if use_mock:
        indices = generate_mock_index_data()
    else:
        indices = fetch_index_data()
        if not indices:
            indices = generate_mock_index_data()
    return snaps, indices


# 将持仓转为可哈希的缓存 key（tuple of tuples）
holdings_raw = st.session_state.get("fund_holdings", load_holdings())
holdings_cache_key = tuple(
    (f["name"], f["code"], f["sector"]) for f in holdings_raw
)

snapshots, indices = load_data(use_mock=use_mock, holdings=holdings_cache_key)

# 数据来源统计
real_count = sum(1 for s in snapshots if s.is_real_data)
mock_count = len(snapshots) - real_count

# 分析报告
industry = analyze_industry_concentration(snapshots)
hedging = analyze_hedging(snapshots)
health = analyze_portfolio_health(snapshots)

# 风险偏好（session_state 持久化）
if "risk_preference" not in st.session_state:
    st.session_state["risk_preference"] = "不确定"

# ============================================================
# 页面标题
# ============================================================

st.title("📊 基金持仓分析系统")

# 数据状态条
if use_mock:
    st.warning("📡 模拟数据模式")
elif mock_count > 0:
    st.error(f"⚠️ {mock_count}/{len(snapshots)} 只基金降级为模拟数据")
else:
    st.success(f"🌐 实时数据 · {real_count}只基金 · 缓存30分钟")

# ============================================================
# 3 个标签页
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 今日看板", "🛡️ 风险分析", "📰 AI简报", "💡 配置建议"])

# ============================================================
# Tab 1: 今日看板
# ============================================================

with tab1:
    # ---- 大盘指数卡片 ----
    st.markdown("### 📈 今日大盘温度")

    if indices:
        cols = st.columns(len(indices))
        for i, idx in enumerate(indices):
            with cols[i]:
                change_pct = idx.daily_change * 100
                arrow = "📈" if change_pct > 0 else ("📉" if change_pct < 0 else "➡️")
                color = "#ef5350" if change_pct > 0 else ("#66bb6a" if change_pct < 0 else "#999")
                st.markdown(f"""
                <div class="index-card {'up' if change_pct > 0 else 'down'}">
                    <div style="font-size:0.9rem; opacity:0.8;">{idx.name}</div>
                    <div style="font-size:1.8rem; font-weight:bold; margin:8px 0;">{idx.price:,.0f}</div>
                    <div style="font-size:1.2rem; color:{color};">{arrow} {change_pct:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("暂无大盘数据")

    # 白话文结论
    if indices:
        up_count = sum(1 for i in indices if i.daily_change > 0)
        if up_count >= 2:
            st.success(
                "📌 今天大盘整体偏暖，三大指数中大部分在涨。"
                "如果持仓中也有基金涨幅较大，可以考虑是否到达了你的止盈点（即你预设的卖出价位）。"
                "反之如果是定投日，涨了也可以少投一点。"
            )
        elif up_count == 0:
            st.error(
                "📌 今天三大指数全部下跌，市场情绪偏冷。"
                "新手看到下跌容易慌，但请记住：基金投资看的是长期（至少半年以上），"
                "一天的涨跌说明不了什么。如果你在定投，今天反而是更好的买入时机——因为同样的钱能买到更多份额。"
            )
        else:
            st.warning(
                "📌 今天市场涨跌互现（即有的涨有的跌），方向不明确。"
                "这种情况下最好的操作就是「不操作」——继续持有，按原计划定投即可。"
            )

    st.markdown("---")

    # ---- 持仓基金表现列表 ----
    st.markdown("### 📋 持仓基金今日表现")

    # 构建表格数据
    rows = []
    for s in snapshots:
        rows.append({
            "基金名称": s.name,
            "代码": s.code,
            "赛道": s.sector,
            "最新净值": f"{s.latest_nav:.4f}",
            "今日涨跌": s.daily_change,
            "近3日": s.change_3d,
            "近5日": s.change_5d,
            "近1周": s.change_1w,
        })

    df = pd.DataFrame(rows)

    # 自定义列配置
    def pct_formatter(val):
        if val > 0:
            return f"📈 +{val*100:.2f}%"
        elif val < 0:
            return f"📉 {val*100:.2f}%"
        return "0.00%"

    df["今日涨跌_str"] = df["今日涨跌"].apply(pct_formatter)
    df["近3日_str"] = df["近3日"].apply(pct_formatter)
    df["近5日_str"] = df["近5日"].apply(pct_formatter)
    df["近1周_str"] = df["近1周"].apply(pct_formatter)

    display_df = df[["基金名称", "代码", "赛道", "最新净值", "今日涨跌_str", "近3日_str", "近5日_str", "近1周_str"]]
    display_df.columns = ["基金名称", "代码", "赛道", "最新净值", "今日涨跌", "近3日", "近5日", "近1周"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=310,
    )

    # ---- 涨跌预警 ----
    st.markdown("##### 🔔 波动预警")

    warnings_found = False
    for s in snapshots:
        # 只对下跌触发预警，上涨不需要警告
        if s.daily_change <= -WARNING_RED:
            st.markdown(f"""
            <div class="warning-red">
                🔴 <b>{s.name}</b>（{s.sector}）下跌 {abs(s.daily_change)*100:.2f}% — 超4%红色预警，建议评估是否减仓或调仓
            </div>""", unsafe_allow_html=True)
            warnings_found = True
        elif s.daily_change <= -WARNING_YELLOW:
            st.markdown(f"""
            <div class="warning-yellow">
                🟡 <b>{s.name}</b>（{s.sector}）下跌 {abs(s.daily_change)*100:.2f}% — 超2.5%黄色预警，关注后续走势
            </div>""", unsafe_allow_html=True)
            warnings_found = True

    if not warnings_found:
        st.success("✅ 今日所有基金波动均在正常范围内")


# ============================================================
# Tab 2: 风险分析
# ============================================================

with tab2:
    st.markdown("### 🛡️ 风险对冲与行业分析")

    # 两栏布局
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🏭 行业集中度")

        # 饼图
        pie_data = pd.DataFrame({
            "赛道": list(industry.sector_counts.keys()),
            "基金数量": list(industry.sector_counts.values()),
        })
        fig_pie = px.pie(pie_data, values="基金数量", names="赛道", hole=0.5)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        # 风险说明
        st.markdown(f"""
        <div style="padding:12px; border-radius:8px; background:{'#fff3cd' if industry.tech_total_pct > 60 else '#d4edda'};">
            <b>{industry.risk_level} 集中度风险</b><br>
            <small>{industry.risk_detail}</small>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### ⚖️ 对冲效果分析")

        # 相关性仪表
        corr = hedging.correlation
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=corr,
            title={"text": "资源 vs 科技 · 相关性"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#1e3a5f"},
                "steps": [
                    {"range": [-1, -0.3], "color": "#4caf50", "name": "有效对冲"},
                    {"range": [-0.3, 0.3], "color": "#ff9800", "name": "弱对冲"},
                    {"range": [0.3, 1], "color": "#f44336", "name": "无对冲"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.8, "value": corr},
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=30, r=30, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 效果卡片
        effect_color = {
            "✅ 有效对冲": "#d4edda",
            "⚠️ 弱对冲": "#fff3cd",
            "❌ 无对冲效果": "#f8d7da",
        }.get(hedging.effectiveness, "#f8f9fa")

        st.markdown(f"""
        <div style="padding:12px; border-radius:8px; background:{effect_color};">
            <b>{hedging.effectiveness}</b><br>
            <small>{hedging.suggestion}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- 波动预警列表 ----
    st.markdown("#### ⚡ 今日波动预警")

    alert_list = []
    for s in snapshots:
        # 只对下跌触发预警
        if s.daily_change <= -WARNING_YELLOW:
            level = "🔴 红色" if s.daily_change <= -WARNING_RED else "🟡 黄色"
            alert_list.append({
                "基金": s.name,
                "赛道": s.sector,
                "今日涨跌": f"{s.daily_change*100:+.2f}%",
                "预警等级": level,
                "近3日": f"{s.change_3d*100:+.2f}%",
                "近1周": f"{s.change_1w*100:+.2f}%",
            })

    if alert_list:
        alert_df = pd.DataFrame(alert_list)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
    else:
        st.success("无异常波动，所有基金运行平稳。")


# ============================================================
# Tab 3: AI简报
# ============================================================

with tab3:
    st.markdown("### 📰 基金经理简报")

    col_btn, col_status = st.columns([3, 7])
    with col_btn:
        generate_btn = st.button(
            "🤖 生成今日简报" if use_ai else "📐 生成规则简报",
            type="primary",
            use_container_width=True,
        )
    with col_status:
        if use_ai:
            st.caption("使用 DeepSeek AI + 联网新闻生成专业分析")
        else:
            st.caption("填入 API Key 可启用 AI 基金经理模式")

    if generate_btn:
        with st.spinner("🤖 AI基金经理正在分析你的持仓..." if use_ai else "📐 正在生成简报..."):
            # 生成简报
            if use_ai:
                briefing = generate_ai_briefing(snapshots, indices, industry, hedging, api_key=ai_key)
            else:
                from data_manager import _generate_rule_briefing
                briefing = _generate_rule_briefing(snapshots, indices, industry, hedging)

        # ---- 简报展示 ----
        st.markdown("---")

        # 1. 市场情绪摘要
        st.markdown("#### 📊 市场情绪摘要")
        st.markdown(f"""<div class="briefing-section">{briefing.market_summary}</div>""", unsafe_allow_html=True)

        # 2. 持仓表现点评
        st.markdown("#### 🏆 持仓表现点评")
        st.markdown(f"""<div class="briefing-section">{briefing.performance_review}</div>""", unsafe_allow_html=True)

        # 3. 风险与操作建议
        st.markdown("#### ⚠️ 风险与操作建议")
        st.markdown(f"""<div class="briefing-section" style="border-left:4px solid #dc3545;">{briefing.risk_and_advice}</div>""", unsafe_allow_html=True)

        # 4. 今日术语
        st.markdown("#### 📚 今日术语学习")
        term = get_random_term()
        st.markdown(f"""
        <div class="term-card">
            <b>💡 {term['term']}</b><br>
            <small>{term['explanation']}</small>
        </div>
        """, unsafe_allow_html=True)

    else:
        # 未点击时显示占位
        st.markdown("---")
        st.info("👆 点击上方按钮生成今日简报")
        st.caption("AI模式：联网搜索行业新闻 + 分析持仓数据，耗时约5-10秒")

    # ---- 术语词汇表 ----
    st.markdown("---")
    with st.expander("📖 全部术语词汇表"):
        for t in GLOSSARY_TERMS:
            st.markdown(f"**{t['term']}**：{t['explanation']}")
            st.markdown("---")


# ============================================================
# Tab 4: 配置建议
# ============================================================

with tab4:
    st.markdown("### 💡 智能配置建议")

    # ---- 1. 持仓评分 ----
    st.markdown("#### ⭐ 当前配置评分")

    col_score, col_detail = st.columns([1, 2])
    with col_score:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:#f8f9fa; border-radius:16px;">
            <div style="font-size:2.5rem;">{health.star_rating}</div>
            <div style="font-size:1.2rem; font-weight:bold; margin-top:8px;">{health.star_count} / 5 星</div>
        </div>
        """, unsafe_allow_html=True)

    with col_detail:
        st.markdown(f"""
        <div style="padding:12px; line-height:2;">
            {health.score_detail}<br>
            <small style="color:#888;">
            💡 理想的配置是：进攻仓 50-60%（博收益）+ 防守仓 20-30%（降波动）+ 避险仓 15-20%（保底）。
            你的当前配置偏进攻，涨时爽但跌时也很痛。
            </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- 2. 健康度诊断 ----
    st.markdown("#### 🩺 持仓健康度诊断")

    if health.diagnosis:
        for i, d in enumerate(health.diagnosis, 1):
            st.markdown(f"""
            <div style="padding:10px 16px; margin:6px 0; background:#fff; border-left:4px solid {'#dc3545' if '🔴' in d else '#ffc107' if '🟡' in d else '#3f51b5'}; border-radius:4px;">
                <b>问题{i}：</b>{d.replace('🔴 ','').replace('🟡 ','').replace('💡 ','')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ 你的持仓结构非常健康！继续保持。")

    st.markdown("---")

    # ---- 3. 推荐调整方案 ----
    st.markdown("#### 📌 推荐调整方案")

    # 风险偏好选择
    radio_options = ["不确定", "我偏好稳健", "我偏好进取"]
    # Map stored value back to radio label (stored as "稳健"/"进取"/"不确定")
    reverse_map = {"稳健": "我偏好稳健", "进取": "我偏好进取", "不确定": "不确定"}
    current_value = st.session_state.get("risk_preference", "不确定")
    radio_default = reverse_map.get(current_value, "不确定")
    pref = st.radio(
        "先告诉我你的风格偏好（选完后建议会变）：",
        radio_options,
        horizontal=True,
        index=radio_options.index(radio_default),
        key="risk_pref_radio",
    )
    # 映射为简短值存储
    pref_map = {"我偏好稳健": "稳健", "我偏好进取": "进取", "不确定": "不确定"}
    st.session_state["risk_preference"] = pref_map[pref]

    recs = generate_recommendations(snapshots, health, st.session_state["risk_preference"])

    if recs:
        for i, r in enumerate(recs, 1):
            op_color = {
                "🟢 买入": "#d4edda",
                "🟢 买入（小仓位）": "#d4edda",
                "🔴 考虑卖出": "#f8d7da",
                "🔴 暂停买入": "#fff3cd",
                "🟡 观察": "#e8eaf6",
                "🟡 持有不动": "#e8eaf6",
            }.get(r["操作"], "#f8f9fa")

            st.markdown(f"""
            <div style="padding:14px 16px; margin:8px 0; background:{op_color}; border-radius:8px; line-height:2;">
                <b>📌 建议{i}：</b>{r['操作']} — <b>{r['基金名称']}</b><br>
                📝 <b>理由：</b>{r['理由']}<br>
                📊 <b>建议比例：</b>{r['建议比例']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无特别建议，当前配置较为合理。")

    st.markdown("---")

    # ---- 4. 小白操作指南 ----
    with st.expander("📖 新手操作指南：怎么买/卖基金？"):
        st.markdown("""
        **买入步骤（以支付宝为例）：**
        1. 打开支付宝 → 搜索框输入基金代码（如 518880）
        2. 点击搜索结果 → 查看基金详情页
        3. 点击「买入」→ 输入金额（建议先用小钱试，比如 50-100 元）
        4. 确认支付 → 完成！

        **卖出步骤：**
        1. 支付宝 → 理财 → 基金 → 我的持仓
        2. 找到要卖的基金 → 点击「卖出」
        3. 输入卖出份额或金额 → 确认

        💡 **新手提示**：买入后不要天天盯着看。基金投资以「月」为单位，短期波动（即涨涨跌跌）是正常的。
        设定一个检查频率（比如每周看一次），然后严格执行。
        """)


# ============================================================
# 页脚
# ============================================================

st.markdown("---")
st.caption(
    f"📊 基金持仓分析系统 v2.0 | "
    f"数据{'模拟中' if use_mock else '来自akshare'} | "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"投资有风险，分析仅供参考"
)
