# ============================================
# 食信通 - 餐饮小微企业信用评估系统
# 基于多源数据融合的智能风控工具
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 页面配置（必须是第一个Streamlit命令）
st.set_page_config(
    page_title="食信通 - 餐饮信用评估系统",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('credit_model.pkl')

@st.cache_data
def load_feature_importance():
    try:
        df = pd.read_csv('feature_importance.csv')
        return df.sort_values('重要性', ascending=True)
    except:
        # 如果没有文件，使用默认值
        return pd.DataFrame({
            '特征': ['冲突指数', '僵尸企业型', '综合信用分', '人均消费', '评分'],
            '重要性': [0.17, 0.10, 0.10, 0.07, 0.07]
        })

# 加载模型和特征重要性
model = load_model()
feature_importance_df = load_feature_importance()

# ============================================
# 页面标题
# ============================================
st.title("🍜 食信通")
st.markdown("#### 基于多源数据融合的餐饮小微企业信用评估系统")
st.markdown("---")

# ============================================
# 侧边栏：数据来源说明
# ============================================
with st.sidebar:
    st.header("📊 数据来源")
    st.markdown("""
    | 数据源 | 字段 |
    |--------|------|
    | 🍔 美团 | 评分、月销量、人均消费、评价数 |
    | 🏢 工商 | 经营年限、注册资本、员工数 |
    | ⚖️ 司法 | 司法案件数 |
    | 📋 处罚 | 行政处罚记录 |
    """)
    st.markdown("---")
    st.caption("基于100家海口餐饮企业真实数据训练")
    st.caption("模型: XGBoost | AUC: 0.606")

# ============================================
# 主区域：输入表单
# ============================================
st.header("📝 企业信息录入")

col1, col2, col3 = st.columns(3)

with col1:
    business_name = st.text_input("🏪 企业名称", placeholder="例如：聚福安·传统老爸茶")
    years = st.number_input("📅 经营年限（年）", min_value=0, max_value=50, value=5, step=1)
    stores = st.number_input("🏠 门店数量", min_value=1, max_value=100, value=1, step=1)

with col2:
    rating = st.slider("⭐ 美团评分", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    monthly_sales = st.number_input("📈 月销量（单）", min_value=0, max_value=10000, value=300, step=50)
    avg_spend = st.number_input("💰 人均消费（元）", min_value=0, max_value=500, value=70, step=10)

with col3:
    review_count = st.number_input("💬 评价数", min_value=0, max_value=10000, value=500, step=50)
    employees = st.number_input("👥 员工数（人）", min_value=0, max_value=500, value=10, step=5)
    has_penalty = st.selectbox("⚠️ 是否有行政处罚", ["无", "有"])

# 高级选项（折叠）
with st.expander("🔧 高级信息（可选）"):
    col_a, col_b = st.columns(2)
    with col_a:
        registered_capital = st.number_input("注册资本（万元）", min_value=0, max_value=10000, value=50, step=10)
        judicial_cases = st.number_input("司法案件数", min_value=0, max_value=50, value=0, step=1)
    with col_b:
        has_business_license = st.selectbox("是否查到工商信息", ["是", "否"])
        data_level = st.selectbox("数据层级", ["门店信息", "总部信息"])

# ============================================
# 评估按钮
# ============================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    evaluate_btn = st.button("🚀 开始评估", type="primary", use_container_width=True)

# ============================================
# 评估逻辑和结果展示
# ============================================
if evaluate_btn:
    if not business_name:
        st.error("请输入企业名称")
        st.stop()
    
    # 处理输入
    penalty_value = 1 if has_penalty == "有" else 0
    has_business_license_value = 1 if has_business_license == "是" else 0
    data_level_value = 0 if data_level == "门店信息" else 1
    
    # 计算衍生特征
    conflict_index = 0
    if rating >= 4.5 and penalty_value == 1:
        conflict_index += 1  # 表面光鲜型
    if rating < 3.5 and years >= 5:
        conflict_index += 1  # 老店口碑差
    if monthly_sales > 300 and judicial_cases >= 1:
        conflict_index += 1  # 高风险高流量型
    if registered_capital > 50 and employees < 10:
        conflict_index += 1  # 空壳嫌疑型
    if has_business_license_value == 1 and employees == 0 and years >= 5:
        conflict_index += 1  # 僵尸企业型
    
    # 交叉特征
    rating_avg_spend = rating * avg_spend
    efficiency = monthly_sales / max(employees, 1)
    consumption_age_ratio = avg_spend / max(years, 1)
    capital_per_person = registered_capital / max(employees, 1)
    legal_risk = penalty_value * judicial_cases
    reputation_speed = review_count / max(years, 1)
    
    # 综合信用分
    comprehensive_score = (
        rating * 20 +
        (1 - penalty_value) * 20 +
        min(years / 20, 1) * 15 +
        min(monthly_sales / 1000, 1) * 15 +
        min(avg_spend / 100, 1) * 10 +
        min(registered_capital / 500, 1) * 10 +
        (1 - min(judicial_cases / 10, 1)) * 10
    )
    
    # 构建输入特征（与训练时的15个特征顺序一致）
    input_features = pd.DataFrame([[
        conflict_index,                                 # 冲突指数
        1 if employees == 0 and years >= 5 else 0,     # 僵尸企业型
        comprehensive_score,                           # 综合信用分
        avg_spend,                                     # 人均消费
        rating,                                        # 评分
        monthly_sales,                                 # 月销量
        registered_capital,                            # 注册资本
        rating_avg_spend,                              # 评分_人均消费
        years,                                         # 经营年限
        consumption_age_ratio,                         # 消费_店龄比
        review_count,                                  # 评价数
        reputation_speed,                              # 口碑积累速度
        has_business_license_value,                    # 是否查到工商
        capital_per_person,                            # 人均资本
        1 if has_business_license_value and employees == 0 and years >= 5 else 0  # 数据源完整度
    ]], columns=[
        '冲突指数', '僵尸企业型', '综合信用分', '人均消费', '评分',
        '月销量', '注册资本(万元)', '评分_人均消费', '经营年限',
        '消费_店龄比', '评价数', '口碑积累速度', '是否查到工商',
        '人均资本', '数据源完整度'
    ])
    
    # 预测
    try:
        default_prob = model.predict_proba(input_features)[0][1]
        credit_score = int(1000 - default_prob * 1000)
    except Exception as e:
        st.warning(f"模型调用出错，使用备用计算: {e}")
        default_prob = 0.3 - (rating - 3) * 0.05 + penalty_value * 0.2
        default_prob = max(0.05, min(0.8, default_prob))
        credit_score = int(1000 - default_prob * 1000)
    
    # 风险等级
    if credit_score >= 800:
        risk_level = "低风险"
        risk_color = "green"
        risk_icon = "🟢"
        interest_rate = "3.5% - 4.5%"
        suggestion = "建议优先推荐，可给予优惠利率"
    elif credit_score >= 600:
        risk_level = "中低风险"
        risk_color = "orange"
        risk_icon = "🟡"
        interest_rate = "5.0% - 6.5%"
        suggestion = "建议正常审批，适当控制额度"
    elif credit_score >= 400:
        risk_level = "中高风险"
        risk_color = "red"
        risk_icon = "🟠"
        interest_rate = "7.0% - 9.0%"
        suggestion = "建议加强尽调，降低授信额度"
    else:
        risk_level = "高风险"
        risk_color = "darkred"
        risk_icon = "🔴"
        interest_rate = "建议拒绝或担保"
        suggestion = "建议拒绝或要求抵押担保"
    
    # ============================================
    # 结果展示
    # ============================================
    st.markdown("---")
    st.header("📊 评估结果")
    
    # 第一行：核心指标
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.metric("信用评分", f"{credit_score} 分", delta=None)
    with col_r2:
        st.markdown(f"### {risk_icon} {risk_level}")
    with col_r3:
        st.metric("违约概率", f"{default_prob*100:.1f}%", delta=None)
    with col_r4:
        st.metric("建议利率", interest_rate, delta=None)
    
    # 第二行：风险因子提示
    st.subheader("⚠️ 风险因子分析")
    
    risk_factors = []
    if penalty_value == 1:
        risk_factors.append("• 存在行政处罚记录，合规风险较高")
    if judicial_cases > 0:
        risk_factors.append(f"• 存在 {judicial_cases} 个司法案件，法律风险较高")
    if rating < 3.5:
        risk_factors.append("• 美团评分偏低，可能影响经营稳定性")
    if years < 2:
        risk_factors.append("• 经营年限较短，抗风险能力待验证")
    if has_business_license_value == 0:
        risk_factors.append("• 未查到工商信息，建议核实企业资质")
    if employees == 0:
        risk_factors.append("• 员工数为0，可能存在经营异常")
    if conflict_index >= 2:
        risk_factors.append(f"• 多源信息存在矛盾（冲突指数={conflict_index}），建议重点关注")
    
    if risk_factors:
        for rf in risk_factors:
            st.warning(rf)
    else:
        st.success("未发现明显风险因子，企业状况良好")
    
    # 第三行：特征重要性图
    st.subheader("📈 特征重要性分析（多源融合特征贡献显著）")
    
    fig = px.bar(
        feature_importance_df.tail(10),
        x='重要性',
        y='特征',
        orientation='h',
        title='模型特征重要性排名',
        color='重要性',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # 第四行：信贷建议
    st.subheader("💡 信贷建议")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.info(f"**评估结论**：{suggestion}")
    with col_s2:
        if credit_score >= 600:
            recommended_amount = max(10, int(credit_score / 50)) * 5
            st.success(f"**建议授信额度**：{recommended_amount} - {recommended_amount * 2} 万元")
        else:
            st.error("**建议授信额度**：暂不建议授信或需担保")
    
    # 第五行：模型说明
    with st.expander("📖 模型说明（点击展开）"):
        st.markdown("""
        **本模型特点：**
        - **多源数据融合**：整合美团、工商、司法、处罚4个数据源
        - **冲突特征设计**：识别“评分高但有处罚”等矛盾信号
        - **可解释性**：输出风险因子和特征重要性，辅助信贷决策
        - **真实数据训练**：基于100家海口餐饮企业真实数据
        
        **模型性能：**
        - AUC: 0.606
        - 特征重要性 Top 3: 冲突指数(17.2%)、僵尸企业型(9.8%)、综合信用分(9.7%)
        """)

# ============================================
# 页脚
# ============================================
st.markdown("---")
st.caption("© 2026 食信通 | 基于多源数据融合的餐饮小微企业信用评估系统 | 数据来源：美团、企查查、信用中国")