"""
data_manager.py — 专业基金持仓分析系统 · 数据层
=================================================
功能：
  1. 基金净值获取（akshare + 模拟降级）
  2. 大盘指数实时行情
  3. 多周期涨跌幅计算
  4. 行业集中度 + 对冲效果分析
  5. DeepSeek AI 基金经理简报
"""

import random
import datetime
import os
import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

USE_MOCK_DATA = False  # 默认真实数据

# 数据文件路径
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdings.json")

# 默认持仓（首次使用或重置时）
DEFAULT_HOLDINGS = [
    {"name": "易方达人工智能ETF联接C", "code": "012734", "sector": "AI/算力"},
    {"name": "易方达资源行业混合",     "code": "110025", "sector": "周期/资源"},
    {"name": "富国新机遇灵活配置混合C", "code": "004675", "sector": "科技+高端制造"},
    {"name": "富国全球科技互联网QDII C","code": "022184", "sector": "全球科技(美股)"},
    {"name": "银河创新成长混合C",       "code": "014143", "sector": "科技成长"},
    {"name": "嘉实上证科创板芯片ETF联接C","code":"017470", "sector": "芯片/半导体"},
    {"name": "永赢高端装备智选混合A",   "code": "015789", "sector": "军工/高端装备"},
]

# 已废弃：保留作为默认兜底
FUND_CONFIG = DEFAULT_HOLDINGS


def load_holdings() -> list[dict]:
    """从 JSON 文件加载持仓列表，失败则返回默认持仓"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            funds = data.get("funds", [])
            if funds and isinstance(funds, list) and all(isinstance(f, dict) and "code" in f for f in funds):
                return funds
    except Exception:
        pass
    return [dict(f) for f in DEFAULT_HOLDINGS]


def save_holdings(funds: list[dict]) -> bool:
    """保存持仓列表到 JSON 文件"""
    try:
        data = {}
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        data["funds"] = funds
        data["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# 大盘指数代码
INDEX_CODES = {
    "上证指数": "000001",
    "沪深300": "000300",
    "科创50":  "000688",
}

# 波动预警阈值
WARNING_YELLOW = 0.025  # 2.5% 黄色预警
WARNING_RED    = 0.04   # 4%   红色预警

# DeepSeek
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 学习术语
GLOSSARY_TERMS = [
    {"term": "最大回撤", "explanation": "在选定周期内，基金净值从最高点跌到最低点的最大跌幅。比如从1.5跌到1.2再涨回去，最大回撤=(1.5-1.2)/1.5=20%。回撤越小，基金经理风控能力越强。"},
    {"term": "夏普比率", "explanation": "衡量每承担1单位风险能获得多少超额回报。夏普>1算优秀，>2非常出色，<0说明不如存银行。选基金时的重要参考指标。"},
    {"term": "贝塔系数", "explanation": "衡量基金相对大盘的波动程度。β=1跟大盘同涨同跌，β>1比大盘更刺激（涨得多跌得也多），β<1更稳健。科技基金通常β>1。"},
    {"term": "阿尔法收益", "explanation": "基金经理靠自身选股能力跑赢市场基准的那部分收益。α>0说明经理厉害，α<0说明不如直接买指数。选主动基金就是为了赚α。"},
    {"term": "市盈率PE", "explanation": "股价÷每股盈利。PE=10理论上10年回本。PE越低通常估值越便宜，但科技股PE天生比银行股高，要跟同行业比。"},
    {"term": "市净率PB", "explanation": "股价÷每股净资产。PB<1叫「破净」，股价比公司净资产还低。常用于评估银行、资源等重资产行业的估值。"},
    {"term": "行业集中度", "explanation": "持仓中同一行业/赛道基金占比。比如你的7只中有5只是科技相关，集中度就很高。集中度高=涨时暴利、跌时暴亏，需要用非相关品种对冲。"},
    {"term": "对冲", "explanation": "配置与主力持仓走势相反或不相关的品种来降低整体波动。比如科技基金+资源/黄金基金，科技跌时资源可能涨，互相抵消部分亏损。"},
]


# ============================================================
# 数据模型
# ============================================================

@dataclass
class FundSnapshot:
    """单只基金快照（聚焦市场表现，不含个人金额）"""
    name: str
    code: str
    sector: str              # 所属赛道
    latest_nav: float        # 最新单位净值
    prev_nav: float          # 前一日单位净值
    daily_change: float      # 今日涨跌幅（小数）
    change_3d: float = 0.0   # 近3日累计涨跌幅
    change_5d: float = 0.0   # 近5日累计涨跌幅
    change_1w: float = 0.0   # 近1周累计涨跌幅
    nav_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    is_real_data: bool = False


@dataclass
class IndexSnapshot:
    """大盘指数快照"""
    name: str
    code: str
    price: float            # 当前点位
    daily_change: float     # 今日涨跌幅（小数）


@dataclass
class IndustryReport:
    """行业集中度分析报告"""
    sector_counts: dict     # {赛道: 基金数量}
    concentration_pct: float  # 最大赛道占比
    risk_level: str         # "高"/"中"/"低"
    risk_detail: str        # 详细风险说明
    tech_total_pct: float   # 科技相关(AI+芯片+科技成长+全球科技)总占比


@dataclass
class HedgingReport:
    """对冲效果分析报告"""
    hedge_fund_name: str    # 对冲品种名称
    correlation: float      # 与科技板块的相关系数（-1~1）
    effectiveness: str      # 对冲效果评价
    suggestion: str         # 对冲建议


@dataclass
class FundBriefing:
    """AI基金经理简报（无个人资产字段）"""
    market_summary: str     # 市场情绪摘要
    performance_review: str # 持仓表现点评
    risk_and_advice: str    # 风险与操作建议
    term_of_day: str        # 今日术语


# ============================================================
# 持久化（API Key）
# ============================================================

def load_api_key() -> str:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("api_key", "")
        except Exception:
            pass
    return ""


def save_api_key(key: str) -> bool:
    try:
        data = {}
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        data["api_key"] = key
        data["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[错误] API Key 保存失败: {e}")
        return False


# ============================================================
# 模拟数据生成
# ============================================================

def _get_base_nav(fund_code: str) -> float:
    """默认基准净值"""
    defaults = {"012734": 2.35, "110025": 2.46, "004675": 2.99,
                "022184": 6.20, "014143": 15.16, "017470": 3.74, "015789": 1.44}
    return defaults.get(fund_code, 1.0)


def generate_mock_nav_history(fund_code: str, days: int = 30) -> pd.DataFrame:
    """生成模拟净值历史"""
    base_nav = _get_base_nav(fund_code)
    today = datetime.date.today()
    dates, navs, current = [], [], base_nav

    for i in range(days, -1, -1):
        date = today - datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue
        if i == days:
            current = base_nav * random.uniform(0.97, 1.03)
        else:
            current = current * (1 + random.uniform(-0.025, 0.025))
        dates.append(date)
        navs.append(round(current, 4))

    df = pd.DataFrame({"date": pd.to_datetime(dates), "nav": navs})
    df["daily_change"] = df["nav"].pct_change().fillna(0)
    return df


def generate_mock_snapshot(fund: dict) -> FundSnapshot:
    """从配置生成模拟快照"""
    code = fund["code"]
    nav_history = generate_mock_nav_history(code, days=30)
    latest = nav_history["nav"].iloc[-1]
    prev = nav_history["nav"].iloc[-2] if len(nav_history) >= 2 else latest
    daily = (latest - prev) / prev if prev != 0 else 0
    multi = _calc_multi_period(nav_history)
    return FundSnapshot(
        name=fund["name"], code=code, sector=fund["sector"],
        latest_nav=round(latest, 4), prev_nav=round(prev, 4),
        daily_change=round(daily, 4),
        change_3d=multi["3d"], change_5d=multi["5d"], change_1w=multi["1w"],
        nav_history=nav_history, is_real_data=False,
    )


# ============================================================
# 多周期涨跌幅计算
# ============================================================

def _calc_multi_period(nav_df: pd.DataFrame) -> dict:
    """从历史净值计算近3日/5日/1周累计涨跌幅"""
    result = {"3d": 0.0, "5d": 0.0, "1w": 0.0}
    if nav_df.empty or len(nav_df) < 2:
        return result
    navs = nav_df["nav"].values
    latest = navs[-1]
    for label, offset in [("3d", 3), ("5d", 5), ("1w", 7)]:
        idx = max(0, len(navs) - 1 - offset)
        if idx < len(navs) - 1:
            base = navs[idx]
            result[label] = round((latest - base) / base, 4) if base != 0 else 0.0
    return result


# ============================================================
# akshare 真实数据
# ============================================================

def fetch_fund_nav_akshare(fund_code: str) -> Optional[dict]:
    """通过 akshare 获取基金净值"""
    try:
        import akshare as ak
        df = None
        for param_name in ["symbol", "fund"]:
            try:
                df = ak.fund_open_fund_info_em(**{param_name: fund_code}, indicator="单位净值走势")
                if df is not None and not df.empty:
                    break
            except TypeError:
                continue
        if df is None or df.empty:
            return None

        date_col = nav_col = change_col = None
        for col in df.columns:
            if "日期" in str(col): date_col = col
            elif "单位净值" in str(col) and "累计" not in str(col): nav_col = col
            elif "日增长" in str(col) or "增长率" in str(col): change_col = col
        if nav_col is None:
            return None

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col, ascending=False)

        latest_nav = float(df[nav_col].iloc[0])
        prev_nav = float(df[nav_col].iloc[1]) if len(df) >= 2 else latest_nav
        nav_change = (latest_nav - prev_nav) / prev_nav if prev_nav != 0 else 0

        if change_col:
            raw_change = float(df[change_col].iloc[0])
            daily_change = _parse_daily_change(raw_change, nav_change)
        else:
            daily_change = nav_change

        # 构建 nav_history（用于多周期计算）
        if date_col:
            hist = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "nav": df[nav_col].astype(float),
            }).dropna().sort_values("date")
            if not hist.empty:
                hist["daily_change"] = hist["nav"].pct_change().fillna(0)
        else:
            hist = pd.DataFrame()

        multi = _calc_multi_period(hist)
        print(f"[数据] {fund_code} 净值={latest_nav:.4f} 日涨跌={daily_change*100:+.2f}%")

        return {
            "latest_nav": latest_nav,
            "prev_nav": prev_nav,
            "daily_change": daily_change,
            "nav_history": hist,
            "multi": multi,
        }
    except ImportError:
        return None
    except Exception as e:
        print(f"[警告] {fund_code} 获取失败: {e}")
        return None


def _parse_daily_change(raw: float, nav_change: float) -> float:
    """交叉验证解析日涨跌幅格式"""
    d1, d2 = abs(raw - nav_change), abs(raw / 100.0 - nav_change)
    return raw / 100.0 if d2 < d1 else raw


def fetch_index_data() -> list[IndexSnapshot]:
    """获取大盘指数实时行情（上证/沪深300/科创50）"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_spot_em()
        if df is None or df.empty:
            return []

        # 列顺序固定：序号/代码/名称/最新价/涨跌幅/涨跌额/...
        cols = list(df.columns)
        code_col = cols[1]   # 代码
        price_col = cols[3]  # 最新价
        change_col = cols[4] # 涨跌幅（百分比数值，如1.16=1.16%）

        results = []
        for name, code in INDEX_CODES.items():
            try:
                row = df[df[code_col].astype(str).str.strip() == code]
                if row.empty:
                    row = df[df[code_col].astype(str).str.contains(code)]
                if row.empty:
                    continue
                row = row.iloc[0]
                price = float(row[price_col])
                chg_raw = float(row[change_col])
                # 百分比形式（1.16 → 0.0116）
                chg = chg_raw / 100.0
                results.append(IndexSnapshot(name=name, code=code, price=price, daily_change=round(chg, 4)))
            except Exception:
                continue
        return results
    except ImportError:
        return []
    except Exception as e:
        print(f"[警告] 指数数据获取失败: {e}")
        return []


def generate_mock_index_data() -> list[IndexSnapshot]:
    """模拟大盘指数数据"""
    return [
        IndexSnapshot("上证指数", "000001", 3350 + random.uniform(-50, 50), round(random.uniform(-0.02, 0.02), 4)),
        IndexSnapshot("沪深300", "000300", 4920 + random.uniform(-60, 60), round(random.uniform(-0.02, 0.02), 4)),
        IndexSnapshot("科创50",  "000688", 2120 + random.uniform(-30, 30), round(random.uniform(-0.03, 0.03), 4)),
    ]


# ============================================================
# 综合数据获取
# ============================================================

def get_all_snapshots(use_mock: bool = None, holdings: list[dict] = None) -> list[FundSnapshot]:
    """获取所有持仓基金快照"""
    if use_mock is None:
        use_mock = USE_MOCK_DATA
    if holdings is None:
        holdings = load_holdings()
    snapshots = []
    for fund in holdings:
        code = fund["code"]
        if use_mock:
            snapshots.append(generate_mock_snapshot(fund))
        else:
            real = fetch_fund_nav_akshare(code)
            if real:
                snapshots.append(FundSnapshot(
                    name=fund["name"], code=code, sector=fund["sector"],
                    latest_nav=round(real["latest_nav"], 4),
                    prev_nav=round(real["prev_nav"], 4),
                    daily_change=round(real["daily_change"], 4),
                    change_3d=real["multi"]["3d"], change_5d=real["multi"]["5d"], change_1w=real["multi"]["1w"],
                    nav_history=real["nav_history"], is_real_data=True,
                ))
            else:
                snapshots.append(generate_mock_snapshot(fund))
    return snapshots


# ============================================================
# 行业集中度分析
# ============================================================

def analyze_industry_concentration(snapshots: list[FundSnapshot]) -> IndustryReport:
    """分析行业集中度，识别风险"""
    # 按赛道统计
    sector_counts = {}
    for s in snapshots:
        sector_counts[s.sector] = sector_counts.get(s.sector, 0) + 1

    total = len(snapshots)
    max_sector = max(sector_counts, key=sector_counts.get) if sector_counts else "未知"
    max_pct = sector_counts[max_sector] / total * 100 if total > 0 else 0

    # 科技相关赛道
    tech_sectors = ["AI/算力", "芯片/半导体", "科技成长", "全球科技(美股)", "科技+高端制造"]
    tech_count = sum(sector_counts.get(ts, 0) for ts in tech_sectors)
    tech_pct = tech_count / total * 100 if total > 0 else 0

    # 风险等级
    if tech_pct >= 80:
        risk_level, risk_detail = "🔴 高", (
            f"科技/AI/芯片相关基金占比高达 {tech_pct:.0f}%，属于极度集中的持仓结构。"
            "一旦科技板块回调（如美联储转鹰、AI泡沫担忧、中美科技摩擦），整个账户将面临同步下跌。"
            "建议配置 **债券基金** 或 **资源/黄金ETF** 作为对冲。"
        )
    elif tech_pct >= 60:
        risk_level, risk_detail = "🟡 中", (
            f"科技相关占比约 {tech_pct:.0f}%，有一定集中度风险。"
            "可适当增加消费、医药或固收类品种，降低科技板块波动对账户的冲击。"
        )
    else:
        risk_level, risk_detail = "🟢 低", "行业分布相对均衡，集中度风险可控。"

    return IndustryReport(
        sector_counts=sector_counts,
        concentration_pct=round(max_pct, 1),
        risk_level=risk_level,
        risk_detail=risk_detail,
        tech_total_pct=round(tech_pct, 1),
    )


# ============================================================
# 对冲效果分析
# ============================================================

def analyze_hedging(snapshots: list[FundSnapshot]) -> HedgingReport:
    """分析资源基金(110025)对科技板块的对冲效果"""
    # 找到资源基金
    resource = next((s for s in snapshots if s.code == "110025"), None)
    tech_funds = [s for s in snapshots if s.sector in
                  ["AI/算力", "芯片/半导体", "科技成长", "全球科技(美股)", "科技+高端制造"]]

    if not resource or not tech_funds:
        return HedgingReport(
            hedge_fund_name="暂无",
            correlation=0.0,
            effectiveness="无法计算",
            suggestion="请确保持仓中包含资源基金(110025)和至少一只科技基金",
        )

    # 计算资源基金与科技基金涨跌幅的相关系数
    resource_changes = None
    tech_all_changes = []  # 收集所有科技基金的涨跌幅序列

    if hasattr(resource, 'nav_history') and not resource.nav_history.empty:
        r_hist = resource.nav_history
        # 先确定所有基金的最小公共长度
        lengths = [len(r_hist)]
        for tech in tech_funds:
            if hasattr(tech, 'nav_history') and not tech.nav_history.empty:
                lengths.append(len(tech.nav_history))
        common_len = min(lengths) if lengths else 0

        if common_len >= 5:
            resource_changes = r_hist["daily_change"].values[-common_len:]
            for tech in tech_funds:
                if hasattr(tech, 'nav_history') and not tech.nav_history.empty:
                    t_changes = tech.nav_history["daily_change"].values[-common_len:]
                    tech_all_changes.append(t_changes)

    if resource_changes is not None and len(tech_all_changes) >= 1:
        # 取所有科技基金涨跌幅的均值
        tech_avg = sum(tech_all_changes) / len(tech_all_changes)
        corr = np.corrcoef(resource_changes, tech_avg)[0, 1]
        corr = round(float(corr), 4)
    else:
        # 用当日涨跌做近似估算
        r_chg = resource.daily_change
        t_avg = sum(t.daily_change for t in tech_funds) / len(tech_funds) if tech_funds else 0
        corr = -0.3 if r_chg * t_avg < 0 else 0.3  # 粗略估计
        corr = round(corr, 4)

    if corr < -0.3:
        effectiveness = "✅ 有效对冲"
        suggestion = (
            f"资源基金与科技板块呈 **负相关({corr:.2f})**，对冲效果良好。"
            "当科技板块下跌时，资源基金往往能逆势上涨，有效缓冲回撤。"
            "建议维持当前配置，若科技仓位继续增加，可同步加仓资源品种以保持对冲比例。"
        )
    elif corr < 0.3:
        effectiveness = "⚠️ 弱对冲"
        suggestion = (
            f"资源基金与科技板块相关性为 {corr:.2f}，对冲效果偏弱。"
            "两只基金的走势基本独立但没有明显负相关，极端行情下可能同涨同跌。"
            "可考虑引入 **黄金ETF** 或 **债券基金** 增强对冲效果。"
        )
    else:
        effectiveness = "❌ 无对冲效果"
        suggestion = (
            f"资源基金与科技板块呈 **正相关({corr:.2f})**，无法起到对冲作用。"
            "两只基金大概率同涨同跌，科技大跌时资源也可能跟跌。"
            "建议调仓：减少资源基金比例，换入 **黄金ETF(518880)** 或 **国债ETF(511010)**。"
        )

    return HedgingReport(
        hedge_fund_name=resource.name if resource else "未知",
        correlation=corr,
        effectiveness=effectiveness,
        suggestion=suggestion,
    )


# ============================================================
# 智能配置建议（小白版）
# ============================================================

# 基金类型分类（进攻/防守/避险）
SECTOR_CLASSIFICATION = {
    "AI/算力":          "进攻",
    "芯片/半导体":       "进攻",
    "科技成长":          "进攻",
    "全球科技(美股)":    "进攻",
    "科技+高端制造":     "进攻",
    "军工/高端装备":     "进攻",
    "周期/资源":         "防守",
}

# 推荐基金库（用于智能建议）
RECOMMENDED_FUNDS = {
    "黄金ETF": {
        "code": "518880", "name": "黄金ETF",
        "type": "避险", "reason": "黄金在股市大跌时通常上涨，是天然的「避风港」",
        "suggested_pct": "10%-15%",
    },
    "债券基金": {
        "code": "110037", "name": "易方达纯债债券",
        "type": "避险", "reason": "纯债基金波动极小（即涨跌幅度很小），每年稳定赚2-4%，像「理财版余额宝」",
        "suggested_pct": "15%-20%",
    },
    "红利低波ETF": {
        "code": "512890", "name": "红利低波ETF",
        "type": "防守", "reason": "专门买分红多、波动小的公司股票，长期持有很稳，适合做底仓",
        "suggested_pct": "10%-15%",
    },
    "消费基金": {
        "code": "161725", "name": "招商中证白酒指数",
        "type": "防守", "reason": "白酒是A股长牛板块（即长期上涨的板块），与科技板块走势不同步，可以分散风险",
        "suggested_pct": "5%-10%",
    },
    "医药基金": {
        "code": "003095", "name": "中欧医疗健康混合",
        "type": "防守", "reason": "医药是刚需行业，不受经济周期影响，长期稳定增长",
        "suggested_pct": "5%-10%",
    },
}


@dataclass
class PortfolioHealth:
    """持仓健康度报告"""
    offense_count: int       # 进攻仓数量
    defense_count: int       # 防守仓数量
    safe_haven_count: int    # 避险仓数量
    offense_pct: float       # 进攻仓占比
    defense_pct: float       # 防守仓占比
    safe_haven_pct: float    # 避险仓占比
    star_rating: str         # 星级评分（⭐）
    star_count: int          # 星级数（1-5）
    diagnosis: list[str]     # 诊断问题列表
    score_detail: str        # 评分说明


def analyze_portfolio_health(snapshots: list[FundSnapshot]) -> PortfolioHealth:
    """分析持仓健康度，给出星级评分和诊断"""
    total = len(snapshots)
    offense = [s for s in snapshots if SECTOR_CLASSIFICATION.get(s.sector) == "进攻"]
    defense = [s for s in snapshots if SECTOR_CLASSIFICATION.get(s.sector) == "防守"]
    safe = [s for s in snapshots if SECTOR_CLASSIFICATION.get(s.sector) == "避险"]

    off_pct = len(offense) / total * 100 if total > 0 else 0
    def_pct = len(defense) / total * 100 if total > 0 else 0
    safe_pct = len(safe) / total * 100 if total > 0 else 0

    diagnosis = []
    score = 5  # 起始满分，逐项扣分

    # 诊断1：进攻仓过多
    if off_pct >= 80:
        diagnosis.append(
            f"🔴 你的科技/成长类基金（进攻仓）占比高达 {off_pct:.0f}%，"
            "一旦科技板块突然大跌，你的账户会非常疼。这就像把所有鸡蛋放在一个篮子里。"
        )
        score -= 2
    elif off_pct >= 60:
        diagnosis.append(
            f"🟡 你的进攻仓占比 {off_pct:.0f}%，偏高但不算危险。"
            "建议不要再加仓科技类基金了，先观察一段时间。"
        )
        score -= 1

    # 诊断2：缺少避险资产
    if safe_pct == 0:
        diagnosis.append(
            "🔴 你完全没有配置避险资产（黄金、债券）。"
            "这就像开车没有安全带——平时没事，但急刹车时你会受伤。"
            "建议至少配置 15%-20% 的债券或黄金基金，作为「安全垫」。"
        )
        score -= 2
    elif safe_pct < 10:
        diagnosis.append(
            "🟡 你的避险资产占比偏低，建议增加到 15% 以上。"
        )
        score -= 1

    # 诊断3：对冲效果
    if def_pct < 15:
        diagnosis.append(
            "🟡 你的防守仓（资源/混合型）占比太低，无法有效对冲科技板块的下跌风险。"
            "建议增加一些与科技「不同步」的品种，比如消费或红利基金。"
        )
        score -= 1

    # 诊断4：基金数量
    if total >= 7:
        diagnosis.append(
            "💡 你持有 7 只基金，数量偏多。对于新手，5-6 只足够覆盖主要方向，"
            "太多反而难以管理。可以考虑合并同类型的基金。"
        )

    score = max(1, score)  # 最低1星
    stars = "⭐" * score + "☆" * (5 - score)

    score_detail = (
        f"进攻仓（科技/成长）{off_pct:.0f}%，防守仓（资源/混合）{def_pct:.0f}%，"
        f"避险仓（债券/黄金）{safe_pct:.0f}%。"
    )

    return PortfolioHealth(
        offense_count=len(offense), defense_count=len(defense),
        safe_haven_count=len(safe),
        offense_pct=off_pct, defense_pct=def_pct, safe_haven_pct=safe_pct,
        star_rating=stars, star_count=score,
        diagnosis=diagnosis, score_detail=score_detail,
    )


def generate_recommendations(
    snapshots: list[FundSnapshot],
    health: PortfolioHealth,
    preference: str = "不确定",
) -> list[dict]:
    """
    根据持仓健康度和用户偏好，生成具体可执行的操作建议

    Args:
        snapshots: 持仓快照
        health: 健康度报告
        preference: "稳健" / "进取" / "不确定"

    Returns:
        建议列表，每条包含 {操作, 基金名称, 代码, 理由, 建议比例}
    """
    recommendations = []

    # 检查当前持仓的代码集合
    held_codes = {s.code for s in snapshots}

    if preference == "稳健":
        # 稳健偏好 → 优先推避险+防守
        if health.safe_haven_pct < 10:
            rec = RECOMMENDED_FUNDS["债券基金"]
            if rec["code"] not in held_codes:
                recommendations.append({
                    "操作": "🟢 买入",
                    "基金名称": f"{rec['name']}（{rec['code']}）",
                    "理由": rec["reason"],
                    "建议比例": f"总资金的 {rec['suggested_pct']}",
                })
            rec2 = RECOMMENDED_FUNDS["红利低波ETF"]
            if rec2["code"] not in held_codes:
                recommendations.append({
                    "操作": "🟢 买入",
                    "基金名称": f"{rec2['name']}（{rec2['code']}）",
                    "理由": rec2["reason"],
                    "建议比例": f"总资金的 {rec2['suggested_pct']}",
                })
        # 科技仓位重 → 建议不加仓
        if health.offense_pct >= 60:
            recommendations.append({
                "操作": "🔴 暂停买入",
                "基金名称": "所有科技/AI/芯片类基金",
                "理由": "你的科技仓位已经足够多（即占比超过60%），再加仓就像赌一个方向，风险太高",
                "建议比例": "—",
            })

    elif preference == "进取":
        # 进取偏好 → 推成长+消费医药搭配
        if health.safe_haven_pct < 10:
            rec = RECOMMENDED_FUNDS["黄金ETF"]
            if rec["code"] not in held_codes:
                recommendations.append({
                    "操作": "🟢 买入（小仓位）",
                    "基金名称": f"{rec['name']}（{rec['code']}）",
                    "理由": "即使你喜欢进取风格，也应该留 10% 的黄金作为「压舱石」，跌市时可以卖出黄金补仓科技基金",
                    "建议比例": "总资金的 10%",
                })
        recommendations.append({
            "操作": "🟡 观察",
            "基金名称": "医药基金（003095 中欧医疗健康）",
            "理由": "医药也是成长型赛道，但与科技不同步，可以作为科技之外的第二成长方向",
            "建议比例": "总资金的 5%-10%（先用小钱试试）",
        })

    else:  # 不确定 → 平衡方案
        if health.safe_haven_pct < 10:
            rec = RECOMMENDED_FUNDS["债券基金"]
            if rec["code"] not in held_codes:
                recommendations.append({
                    "操作": "🟢 买入",
                    "基金名称": f"{rec['name']}（{rec['code']}）",
                    "理由": rec["reason"],
                    "建议比例": f"总资金的 {rec['suggested_pct']}",
                })
        if health.defense_pct < 20:
            rec2 = RECOMMENDED_FUNDS["红利低波ETF"]
            if rec2["code"] not in held_codes:
                recommendations.append({
                    "操作": "🟢 买入",
                    "基金名称": f"{rec2['name']}（{rec2['code']}）",
                    "理由": rec2["reason"],
                    "建议比例": f"总资金的 {rec2['suggested_pct']}",
                })
        if health.offense_pct >= 60:
            recommendations.append({
                "操作": "🟡 持有不动",
                "基金名称": "现有科技类基金",
                "理由": "科技仓位已经不低了，暂时不要加仓也不要恐慌卖出。如果你买的基金基本面没问题，跌了反而是加仓机会（这叫「定投」）",
                "建议比例": "—",
            })

    # 如果资源基金没有起到对冲作用，建议替换
    hedge = analyze_hedging(snapshots)
    if "110025" in held_codes and hedge.correlation > 0.3:
        # 检查是否已经有替换建议
        if not any("黄金" in r.get("基金名称", "") for r in recommendations):
            rec = RECOMMENDED_FUNDS["黄金ETF"]
            recommendations.append({
                "操作": "🔴 考虑卖出",
                "基金名称": f"易方达资源行业混合（110025）→ 换入 {rec['name']}（{rec['code']}）",
                "理由": f"资源基金与你的科技基金走势趋同（相关性 {hedge.correlation:.2f}），没有起到对冲作用。换成黄金ETF效果更好。",
                "建议比例": f"总资金的 {rec['suggested_pct']}",
            })

    return recommendations


# ============================================================
# 实时市场感知模块（v2.1 升级）
# ============================================================

@dataclass
class MarketTheme:
    """市场主线主题"""
    name: str           # 板块名称
    change_pct: float   # 涨跌幅
    is_hot: bool        # 是否热点板块


@dataclass
class FundManagerInfo:
    """基金经理信息"""
    name: str
    fund_name: str
    experience_years: str  # 从业年限
    manage_duration: str   # 管理本基金时长
    return_during_tenure: str  # 任职回报
    aum: str              # 管理规模


def fetch_market_themes() -> list[MarketTheme]:
    """获取今日市场热点板块（概念板块涨幅排名）"""
    try:
        import akshare as ak
        df = ak.stock_board_concept_name_em()
        if df is None or df.empty:
            return []
        # 按涨跌幅降序排列，取前5
        cols = list(df.columns)
        name_col = cols[1] if len(cols) > 1 else None
        change_col = cols[5] if len(cols) > 5 else None
        if name_col is None or change_col is None:
            return []
        df["_change"] = pd.to_numeric(df[change_col], errors="coerce")
        top = df.nlargest(5, "_change")
        themes = []
        for _, row in top.iterrows():
            chg = float(row[change_col])
            themes.append(MarketTheme(
                name=str(row[name_col]),
                change_pct=round(chg / 100.0 if abs(chg) > 1 else chg, 4),
                is_hot=abs(chg) > 3,
            ))
        return themes
    except Exception as e:
        print(f"[警告] 市场主线获取失败: {e}")
        return []


# 基金经理全量数据缓存（全局，避免重复拉取）
_manager_cache: Optional[pd.DataFrame] = None


def _get_manager_df() -> Optional[pd.DataFrame]:
    """获取全量基金经理数据（只拉取一次）"""
    global _manager_cache
    if _manager_cache is not None:
        return _manager_cache
    try:
        import akshare as ak
        _manager_cache = ak.fund_manager_em()
        return _manager_cache
    except Exception:
        return None


def fetch_fund_manager_info(fund_code: str) -> Optional[FundManagerInfo]:
    """获取基金经理信息（首次调用约10秒，后续从缓存秒出）"""
    try:
        df = _get_manager_df()
        if df is None or df.empty:
            return None
        cols = list(df.columns)
        # 列：序号/姓名/所属公司/现任基金代码/现任基金名称/累计从业时间/现任基金资产总规模/现任基金最佳回报
        code_col = next((c for c in cols if "代码" in str(c)), None)
        name_col = next((c for c in cols if "姓名" in str(c)), None)
        if code_col is None:
            return None
        match = df[df[code_col].astype(str).str.strip() == fund_code]
        if match.empty:
            return None
        row = match.iloc[0]
        return FundManagerInfo(
            name=str(row[name_col]) if name_col else "未知",
            fund_name=fund_code,
            experience_years=f"{row.iloc[5]}天" if len(row) > 5 else "未知",
            manage_duration="—",
            return_during_tenure=f"{row.iloc[7]}%" if len(row) > 7 and pd.notna(row.iloc[7]) else "未知",
            aum=f"{row.iloc[6]}亿" if len(row) > 6 and pd.notna(row.iloc[6]) else "未知",
        )
    except Exception as e:
        print(f"[调试] 基金经理{fund_code}获取失败: {e}")
        return None


def fetch_reference_fund_performance(codes: list[str]) -> dict:
    """批量获取参考基金的近期表现（用于推荐时的市场状态）"""
    result = {}
    for code in codes:
        data = fetch_fund_nav_akshare(code)
        if data:
            result[code] = {
                "latest_nav": data["latest_nav"],
                "daily_change": data["daily_change"],
            }
    return result


def generate_market_aware_recommendations(
    snapshots: list[FundSnapshot],
    health: PortfolioHealth,
    themes: list[MarketTheme],
    api_key: str = None,
    preference: str = "不确定",
) -> list[dict]:
    """
    基于实时市场状态的智能推荐（v2.1）
    - 联网获取市场热点
    - 获取参考基金实时表现
    - AI 分析后给出诚实建议（不推荐正在下跌的资产）
    """
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        # 无 AI 时降级到规则推荐
        return generate_recommendations(snapshots, health, preference)

    # 获取参考基金的实时表现
    ref_codes = ["518880", "110037", "512890", "161725", "003095", "511010", "589130"]
    ref_perf = fetch_reference_fund_performance(ref_codes)

    # 构建市场主线文本
    theme_text = ""
    if themes:
        hot_names = [t.name for t in themes[:3]]
        theme_text = f"今日市场热点板块（涨幅前三）：{', '.join(hot_names)}。"

    # 构建参考基金表现文本
    ref_text = ""
    for code, perf in ref_perf.items():
        ref_text += f"{code}：净值{perf['latest_nav']:.4f}，今日{perf['daily_change']*100:+.2f}%\n"

    # 构建持仓文本
    holding_text = ""
    for s in snapshots:
        holding_text += (
            f"{s.name}（{s.code}，{s.sector}）："
            f"今日{s.daily_change*100:+.2f}%，近3日{s.change_3d*100:+.2f}%，"
            f"近1周{s.change_1w*100:+.2f}%\n"
        )

    # AI prompt
    system_prompt = """你是一位专业的基金投资顾问，拥有10年经验。用户是基金新手。

## 核心原则：诚实反映市场状态
- 绝对不能为了"配置均衡"而推荐正在大跌的资产
- 如果某个品种近期一直在跌，必须如实告诉用户"现在不适合买入"
- 如果用户持仓与市场主线一致，鼓励持有而不是盲目调仓
- 每条建议必须包含：当前市场状态 + 为什么现在适合/不适合 + 具体操作

## 输出格式
JSON数组，每条建议包含：
- action: 操作（买入/卖出/持有/关注/暂缓）
- fund_name: 基金名称+代码
- market_state: 该品种当前市场状态（1句话：近1月涨跌、资金流向、是否主线）
- reason: 推荐理由（为什么现在适合/不适合，结合实时数据，说白话）
- risk: 当前买入的最大风险是什么
- suggested_pct: 建议比例（如"10%"或"—"）
- priority: 优先级（high/medium/low）"""

    user_prompt = f"""用户风险偏好：{preference}

{theme_text}

用户当前持仓：
{holding_text}
健康度：进攻仓{health.offense_pct:.0f}%，防守仓{health.defense_pct:.0f}%，避险仓{health.safe_haven_pct:.0f}%

备选参考基金今日表现：
{ref_text}

请基于以上实时数据，给出3-4条市场感知的投资建议（JSON数组格式）。
要求：如果备选基金中某个品种今日大跌，不要推荐它，如实说明原因并给出替代方案。"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        # AI may return {"recommendations": [...]} or just [...]
        if isinstance(data, dict):
            recs = data.get("recommendations", [data])
        else:
            recs = data if isinstance(data, list) else []
        return recs
    except Exception as e:
        print(f"[警告] AI推荐生成失败: {e}")
        return generate_recommendations(snapshots, health, preference)


# ============================================================
# 金融新闻获取（用于AI简报的联网信息）
# ============================================================

def fetch_financial_news(keywords: list[str]) -> str:
    """通过 akshare 获取金融新闻摘要（作为AI简报的素材）"""
    news_items = []
    try:
        import akshare as ak
        # 尝试获取财联社电报
        try:
            df = ak.stock_info_global_em()
            if df is not None and not df.empty:
                # 取最新10条标题
                if "标题" in df.columns:
                    titles = df["标题"].head(10).tolist()
                    news_items.extend(str(t) for t in titles)
        except Exception:
            pass

        # 尝试获取东方财富新闻
        try:
            df = ak.stock_zh_a_alerts_cls()
            if df is not None and not df.empty:
                for kw in keywords:
                    mask = df.apply(lambda row: any(kw in str(v) for v in row.values), axis=1)
                    matched = df[mask].head(3)
                    for _, row in matched.iterrows():
                        news_items.append(str(row.iloc[0]) if len(row) > 0 else "")
        except Exception:
            pass
    except ImportError:
        pass

    if news_items:
        return "；".join(news_items[:15])  # 最多15条
    return "暂无行业新闻数据"


# ============================================================
# AI 基金经理简报
# ============================================================

# 专业基金经理 System Prompt
FUND_MANAGER_SYSTEM_PROMPT = """你是一位专业的基金投资顾问，拥有10年资产配置经验。
你的用户是基金新手，刚接触基金不久。
你的职责是帮助用户理解市场动态，识别持仓风险，并提供理性的、数据驱动的操作建议。

## 语言风格要求（强制执行）
1. **禁止单独罗列数据**：不允许只写"科创50涨了4.61%"就结束。每个数据后面必须跟一句白话文结论，用📌或💡开头。
2. **禁止使用未解释的术语**：如果必须用到"正相关"、"回撤"、"夏普比率"等专业词，必须加括号用大白话注解，例如"正相关（即同涨同跌的意思）"。
3. **所有建议必须可执行**：不能说"考虑降低风险"，要说"建议卖出XXX，换入XXX，操作步骤：在支付宝搜索代码，点击买入"。必须给出具体基金代码。
4. **用颜色和图标辅助理解**：红色=坏消息（跌了、风险高），绿色=好消息（涨了、健康），黄色=需要注意。

## 输出格式
JSON格式，包含以下字段：
- market_summary: 市场情绪摘要（2-3句话，白话文+数据解读，不要只列数据）
- performance_review: 持仓表现点评（分析最强/最弱基金+白话原因，结合行业新闻）
- risk_and_advice: 风险与操作建议（指出风险+给出具体基金代码的操作方案，不说空话）
- term_of_day: 从最大回撤/夏普比率/贝塔系数/阿尔法收益/市盈率PE/行业集中度/对冲中选一个，用白话解释（术语名+一句大白话解释，如"最大回撤就是你这只基金最多能亏多少的意思"）"""


def generate_ai_briefing(
    snapshots: list[FundSnapshot],
    indices: list[IndexSnapshot],
    industry: IndustryReport,
    hedging: HedgingReport,
    api_key: str = None,
) -> FundBriefing:
    """AI 生成基金经理简报（含联网新闻）"""
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        return _generate_rule_briefing(snapshots, indices, industry, hedging)

    # 构建持仓数据文本
    fund_text = ""
    for i, s in enumerate(snapshots, 1):
        fund_text += (
            f"{i}. {s.name}（{s.sector}）：净值{s.latest_nav:.4f}，"
            f"今日{s.daily_change*100:+.2f}%，近3日{s.change_3d*100:+.2f}%，"
            f"近5日{s.change_5d*100:+.2f}%，近1周{s.change_1w*100:+.2f}%\n"
        )

    # 大盘数据
    index_text = "\n".join(
        f"{idx.name}：{idx.price:.2f}点，今日{idx.daily_change*100:+.2f}%"
        for idx in indices
    ) if indices else "无大盘数据"

    # 联网获取行业新闻（针对最大涨跌基金所属赛道）
    keywords = list(set(s.sector.split("/")[0] for s in snapshots))
    news_text = fetch_financial_news(keywords)

    # 集中度
    concentration_text = (
        f"科技相关占比{industry.tech_total_pct:.0f}%，"
        f"集中度风险等级：{industry.risk_level}。{industry.risk_detail}"
    )

    # 对冲
    hedge_text = (
        f"对冲品种：{hedging.hedge_fund_name}，"
        f"与科技板块相关性{hedging.correlation:.2f}，"
        f"效果：{hedging.effectiveness}。{hedging.suggestion}"
    )

    user_prompt = f"""以下是今日市场数据和持仓信息，请生成专业简报。

【大盘指数】
{index_text}

【持仓基金表现】
{fund_text}

【行业新闻（联网获取）】
{news_text}

【行业集中度分析】
{concentration_text}

【对冲效果分析】
{hedge_text}

请生成今日基金经理简报（JSON格式）。"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": FUND_MANAGER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return FundBriefing(
            market_summary=data.get("market_summary", "今日市场平稳运行。"),
            performance_review=data.get("performance_review", "暂无点评。"),
            risk_and_advice=data.get("risk_and_advice", "请保持纪律，坚持长期策略。"),
            term_of_day=data.get("term_of_day", "最大回撤：在一定周期内基金净值从最高点到最低点的最大跌幅。"),
        )
    except ImportError:
        return _generate_rule_briefing(snapshots, indices, industry, hedging)
    except Exception as e:
        print(f"[警告] AI简报生成失败: {e}")
        return _generate_rule_briefing(snapshots, indices, industry, hedging)


def _generate_rule_briefing(
    snapshots: list[FundSnapshot],
    indices: list[IndexSnapshot],
    industry: IndustryReport,
    hedging: HedgingReport,
) -> FundBriefing:
    """规则引擎版简报（降级方案），遵循白话文+可执行规则"""
    best = max(snapshots, key=lambda s: s.daily_change)
    worst = min(snapshots, key=lambda s: s.daily_change)
    avg_change = sum(s.daily_change for s in snapshots) / len(snapshots) if snapshots else 0

    # 大盘指数白话解读
    if indices:
        up_count = sum(1 for i in indices if i.daily_change > 0)
        if up_count >= 2:
            idx_note = "大部分指数在涨，市场情绪偏暖"
        elif up_count == 0:
            idx_note = "三大指数全跌，市场情绪偏冷——如果你在定投，今天反而是更好的买入时机（同样的钱能买更多份额）"
        else:
            idx_note = "涨跌互现（即有的涨有的跌），方向不明确，最好的操作就是不操作"
    else:
        idx_note = "暂无大盘数据"

    if avg_change > 0.005:
        market = (
            f"📌 {idx_note}。你的持仓基金今天平均上涨 {avg_change*100:+.2f}%，"
            f"其中 {best.name}（{best.sector}）表现最好。"
        )
    elif avg_change < -0.005:
        market = (
            f"📌 {idx_note}。你的持仓基金今天平均下跌 {abs(avg_change)*100:+.2f}%，"
            f"其中 {worst.name}（{worst.sector}）跌幅最大。"
            "不过一天的下跌不代表什么，基金投资看的是长期（至少半年以上）。"
        )
    else:
        market = f"📌 {idx_note}。你的持仓基金今天涨跌各半，整体变化不大。"

    # 持仓点评
    performance = (
        f"📌 今天 {best.name} 涨 {best.daily_change*100:+.2f}%，"
        f"在 {best.sector} 赛道里表现不错。"
        f"而 {worst.name} 跌了 {abs(worst.daily_change)*100:+.2f}%，"
        f"属于 {worst.sector}，该板块近期波动较大。"
    )

    # 风险与可执行建议
    if worst.daily_change <= -WARNING_RED:
        action = (
            f"如果你仓位较重（超过总资金20%在一只基金上），可以考虑卖出1/3，"
            f"换入债券基金（110037 易方达纯债），降低整体波动。"
        )
        risk = (
            f"⚠️ {worst.name} 今日跌幅超4%（跌了{abs(worst.daily_change)*100:.2f}%），触发红色预警。"
            f"当前你的科技类基金占比{industry.tech_total_pct:.0f}%，"
            f"属于高度集中——涨的时候爽，跌的时候也会很疼。{action}"
        )
    elif worst.daily_change <= -WARNING_YELLOW:
        risk = (
            f"⚡ {worst.name} 今日跌幅超2.5%，黄色预警。"
            f"先别急着卖，观察接下来2-3天走势再决定。"
            f"如果连续下跌，可以考虑把总资金的10%转入黄金ETF（518880）作为对冲（即用黄金的上涨来抵消科技的下跌）。"
        )
    else:
        tips = [
            "可以打开支付宝搜索 110037（易方达纯债），用总资金的10%建立债券底仓",
            "如果你还没设止盈点（即涨到多少就卖），建议设为15%-20%",
            "今天波动不大，维持现有仓位不动就是最好的操作",
        ]
        risk = (
            f"今日无显著波动预警。你的科技占比{industry.tech_total_pct:.0f}%，"
            f"对冲品种 {hedging.hedge_fund_name} 与科技的相关性为{hedging.correlation:.2f}"
            f"（{'同涨同跌' if hedging.correlation > 0 else '反向波动'}）。"
            f"💡 {random.choice(tips)}。"
        )

    term = random.choice(GLOSSARY_TERMS)
    term_text = f"{term['term']}：{term['explanation'][:100]}"

    return FundBriefing(
        market_summary=market,
        performance_review=performance,
        risk_and_advice=risk,
        term_of_day=term_text,
    )


def get_random_term() -> dict:
    return random.choice(GLOSSARY_TERMS)


def generate_industry_explainer(industry_name: str, api_key: str = None) -> str:
    """生成行业科普卡片（AI优先，规则降级）"""
    key = api_key or DEEPSEEK_API_KEY
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是行业研究专家。用大白话解释一个行业/概念板块，面向基金新手。输出不要超过400字。"},
                    {"role": "user", "content": f"请用大白话介绍「{industry_name}」这个板块：1)它是做什么的 2)涨跌受什么影响 3)适合长期持有还是短期炒作 4)当前机构怎么看"},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception:
            pass

    # 规则降级
    return f"""
**{industry_name}** 是一个A股概念板块。

💡 **它是做什么的**：该板块包含与「{industry_name}」相关的上市公司。概念板块通常基于某个主题或政策热点形成。

📊 **涨跌受什么影响**：概念板块的涨跌通常受政策消息、市场情绪和资金面驱动。短期波动可能较大。

⚠️ **适合长期还是短期**：概念板块更适合短期交易，不太适合长期持有。如果你不了解这个行业，建议先学习再决定是否参与。

📌 **当前机构观点**：暂无最新研报数据。建议通过东方财富或同花顺搜索「{industry_name} 研报」查看机构最新观点。
"""

# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据模块自测")
    print("=" * 60)

    snaps = get_all_snapshots(use_mock=True)
    for s in snaps:
        print(f"{s.name}: 日{s.daily_change*100:+.2f}% 3日{s.change_3d*100:+.2f}% 5日{s.change_5d*100:+.2f}% 1周{s.change_1w*100:+.2f}%")

    idx = generate_mock_index_data()
    for i in idx:
        print(f"{i.name}: {i.price:.2f} {i.daily_change*100:+.2f}%")

    ind = analyze_industry_concentration(snaps)
    print(f"集中度: 科技{ind.tech_total_pct}% {ind.risk_level}")

    hedge = analyze_hedging(snaps)
    print(f"对冲: {hedge.hedge_fund_name} 相关性{hedge.correlation} {hedge.effectiveness}")

    brief = _generate_rule_briefing(snaps, idx, ind, hedge)
    print(f"简报: {brief.market_summary}")
    print("全部通过")
