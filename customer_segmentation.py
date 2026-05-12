
---

## **FILE 2: `customer_segmentation.py`**

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

print("="*60)
print("🎯 CUSTOMER SEGMENTATION PROJECT")
print("="*60)

# ============================================
# 1. GENERATE CUSTOMER DATA
# ============================================
print("\n📁 Step 1: Generating customer data...")

random.seed(42)
np.random.seed(42)

# Customer demographics
first_names = ["John","Jane","Mike","Sarah","David","Emma","Chris","Lisa","Raj","Priya","Tom","Amy","Alex","Maria","James","Linda","Robert","Patricia","Michael","Jennifer"]
last_names = ["Smith","Doe","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Wilson","Kumar","Sharma","Patel","Singh","Lee","Wong","Kim","Park","Gupta","Mehta"]
regions = ["North","South","East","West","Central"]
genders = ["Male","Female","Other"]

# Generate customer base
num_customers = 500
customers = []

for i in range(1, num_customers + 1):
    # Demographics
    age = random.randint(18, 70)
    region = random.choice(regions)
    gender = random.choice(genders)
    
    # Behavioral patterns (correlated with age and region)
    if age < 30:
        frequency = random.randint(5, 30)  # Young: shop often
        avg_spend = random.randint(500, 3000)
    elif age < 50:
        frequency = random.randint(3, 20)  # Middle: moderate
        avg_spend = random.randint(1000, 5000)
    else:
        frequency = random.randint(1, 12)  # Senior: less often
        avg_spend = random.randint(800, 4000)
    
    # Region adjustments
    if region in ["North","West"]:
        avg_spend = int(avg_spend * 1.2)  # Higher spending regions
    
    # Calculate metrics
    total_spend = frequency * avg_spend
    recency = random.randint(1, 90)  # Days since last purchase
    categories = random.sample(["Electronics","Clothing","Books","Home","Sports"], 
                               random.randint(1, 4))
    
    # Preference scores (1-10)
    electronics_pref = random.randint(1, 10) if "Electronics" in categories else random.randint(1, 5)
    clothing_pref = random.randint(1, 10) if "Clothing" in categories else random.randint(1, 5)
    books_pref = random.randint(1, 10) if "Books" in categories else random.randint(1, 5)
    home_pref = random.randint(1, 10) if "Home" in categories else random.randint(1, 5)
    sports_pref = random.randint(1, 10) if "Sports" in categories else random.randint(1, 5)
    
    customers.append({
        "customer_id": i,
        "first_name": random.choice(first_names),
        "last_name": random.choice(last_names),
        "age": age,
        "gender": gender,
        "region": region,
        "total_spend": total_spend,
        "frequency": frequency,
        "avg_order_value": avg_spend,
        "recency_days": recency,
        "electronics_score": electronics_pref,
        "clothing_score": clothing_pref,
        "books_score": books_pref,
        "home_score": home_pref,
        "sports_score": sports_pref,
        "tenure_months": random.randint(1, 60)
    })

df = pd.DataFrame(customers)
df.to_csv("customer_data.csv", index=False)
print(f"✅ Generated {len(df)} customers")

# ============================================
# 2. FEATURE ENGINEERING
# ============================================
print("\n🔧 Step 2: Engineering features for clustering...")

# Create RFM-style features
df["spend_per_visit"] = df["total_spend"] / df["frequency"]
df["loyalty_score"] = df["frequency"] / (df["recency_days"] / 30)  # Frequency vs recency
df["avg_category_score"] = df[["electronics_score","clothing_score","books_score","home_score","sports_score"]].mean(axis=1)

# Select features for clustering
features = ["total_spend", "frequency", "avg_order_value", "recency_days", 
            "avg_category_score", "tenure_months", "spend_per_visit", "loyalty_score"]

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# ============================================
# 3. K-MEANS CLUSTERING
# ============================================
print("\n🤖 Step 3: Performing K-Means clustering...")

# Find optimal K using Elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)

# Choose K=4 (optimal balance)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(scaled_features)

# Name segments based on characteristics
segment_names = {
    0: "VIP Customers",
    1: "Regular Shoppers", 
    2: "Occasional Buyers",
    3: "New Customers"
}

# Analyze segment characteristics to assign names
segment_profiles = df.groupby("segment")[["total_spend", "frequency", "recency_days", "tenure_months"]].mean()
print("\n📊 Segment Profiles:")
print(segment_profiles)

# Rename segments intelligently
segment_mapping = {}
for seg in df["segment"].unique():
    profile = segment_profiles.loc[seg]
    if profile["total_spend"] > 50000 and profile["frequency"] > 15:
        segment_mapping[seg] = "💎 VIP Customers"
    elif profile["frequency"] > 8:
        segment_mapping[seg] = "🔄 Regular Shoppers"
    elif profile["tenure_months"] < 6:
        segment_mapping[seg] = "🆕 New Customers"
    else:
        segment_mapping[seg] = "📆 Occasional Buyers"

df["segment_name"] = df["segment"].map(segment_mapping)

# Save results
df.to_csv("segmentation_results.csv", index=False)
print(f"✅ Customers segmented into {optimal_k} groups")

# ============================================
# 4. CALCULATE SEGMENT METRICS
# ============================================
print("\n📈 Step 4: Calculating segment analytics...")

segment_stats = df.groupby("segment_name").agg({
    "customer_id": "count",
    "total_spend": ["mean", "sum"],
    "frequency": "mean",
    "avg_order_value": "mean",
    "recency_days": "mean",
    "tenure_months": "mean"
}).round(2)

segment_stats.columns = ["customer_count", "avg_spend", "total_revenue", "avg_frequency", 
                         "avg_order_value", "avg_recency_days", "avg_tenure_months"]
segment_stats = segment_stats.sort_values("total_revenue", ascending=False)

print("\n" + "="*60)
print("SEGMENT PERFORMANCE SUMMARY")
print("="*60)
for segment in segment_stats.index:
    print(f"\n{segment}")
    print(f"  Customers: {int(segment_stats.loc[segment, 'customer_count'])}")
    print(f"  Total Revenue: ₹{segment_stats.loc[segment, 'total_revenue']:,.2f}")
    print(f"  Avg Spend: ₹{segment_stats.loc[segment, 'avg_spend']:,.2f}")
    print(f"  Frequency: {segment_stats.loc[segment, 'avg_frequency']:.1f} purchases")
    print(f"  Recency: {segment_stats.loc[segment, 'avg_recency_days']:.0f} days ago")

# ============================================
# 5. CREATE VISUALIZATIONS
# ============================================
print("\n📊 Step 5: Creating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Segment Distribution (Pie Chart)
fig1, ax1 = plt.subplots(figsize=(10, 6))
segment_counts = df["segment_name"].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Customer Segment Distribution', fontsize=16, fontweight='bold')
plt.setp(autotexts, size=10, weight="bold", color="white")
plt.tight_layout()
plt.savefig('segment_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 2: Segment Characteristics (Bar Chart)
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Segment Characteristics Comparison', fontsize=16, fontweight='bold')

metrics = ['avg_spend', 'avg_frequency', 'avg_order_value', 'avg_tenure_months']
titles = ['Average Spend (₹)', 'Purchase Frequency', 'Average Order Value (₹)', 'Customer Tenure (months)']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    row, col = idx // 2, idx % 2
    segment_stats[metric].sort_values().plot(kind='barh', ax=axes[row, col], color=colors)
    axes[row, col].set_title(title, fontsize=12)
    axes[row, col].set_xlabel('')
    axes[row, col].tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig('segment_profiles.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 3: 3D Interactive Visualization
fig3 = px.scatter_3d(df, x='total_spend', y='frequency', z='avg_order_value',
                      color='segment_name', size='tenure_months',
                      hover_data=['customer_id', 'region', 'age'],
                      title='Customer Segments - 3D Visualization',
                      color_discrete_sequence=colors,
                      labels={'total_spend': 'Total Spend (₹)',
                              'frequency': 'Purchase Frequency',
                              'avg_order_value': 'Avg Order Value (₹)'})

fig3.update_layout(
    title_font_size=20,
    scene=dict(
        xaxis_title='Total Spend (₹)',
        yaxis_title='Purchase Frequency',
        zaxis_title='Avg Order Value (₹)'
    ),
    width=1000,
    height=700
)
fig3.write_html("segment_analysis.html")
print("✅ 3D interactive visualization saved: segment_analysis.html")

# Figure 4: Heatmap of Category Preferences by Segment
category_cols = ['electronics_score', 'clothing_score', 'books_score', 'home_score', 'sports_score']
category_avg = df.groupby('segment_name')[category_cols].mean()

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(category_avg, annot=True, cmap='YlOrRd', fmt='.1f', ax=ax4, cbar_kws={'label': 'Preference Score'})
ax4.set_title('Product Category Preferences by Segment', fontsize=14, fontweight='bold')
ax4.set_xlabel('Category')
ax4.set_ylabel('Customer Segment')
plt.tight_layout()
plt.savefig('category_preferences.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 5: Revenue Contribution by Segment
fig5, ax5 = plt.subplots(figsize=(10, 6))
revenue_by_segment = df.groupby('segment_name')['total_spend'].sum().sort_values()
revenue_by_segment.plot(kind='barh', ax=ax5, color=colors)
ax5.set_title('Total Revenue Contribution by Segment', fontsize=14, fontweight='bold')
ax5.set_xlabel('Total Revenue (₹)')
ax5.set_ylabel('Customer Segment')
for i, v in enumerate(revenue_by_segment.values):
    ax5.text(v + 10000, i, f'₹{v:,.0f}', va='center')
plt.tight_layout()
plt.savefig('revenue_contribution.png', dpi=100, bbox_inches='tight')
plt.close()

print("✅ 5 visualizations created")

# ============================================
# 6. BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================
print("\n💡 Step 6: Generating business insights...")

insights = f"""
{'='*60}
CUSTOMER SEGMENTATION INSIGHTS
{'='*60}

📊 KEY FINDINGS:

1. SEGMENT COMPOSITION:
   - VIP Customers: {len(df[df['segment_name']=='💎 VIP Customers'])} customers ({len(df[df['segment_name']=='💎 VIP Customers'])/len(df)*100:.1f}%)
   - Regular Shoppers: {len(df[df['segment_name']=='🔄 Regular Shoppers'])} customers ({len(df[df['segment_name']=='🔄 Regular Shoppers'])/len(df)*100:.1f}%)
   - Occasional Buyers: {len(df[df['segment_name']=='📆 Occasional Buyers'])} customers ({len(df[df['segment_name']=='📆 Occasional Buyers'])/len(df)*100:.1f}%)
   - New Customers: {len(df[df['segment_name']=='🆕 New Customers'])} customers ({len(df[df['segment_name']=='🆕 New Customers'])/len(df)*100:.1f}%)

2. REVENUE CONCENTRATION:
   - Top 20% of customers (VIP) contribute {segment_stats.loc['💎 VIP Customers', 'total_revenue']/df['total_spend'].sum()*100:.1f}% of total revenue
   - VIP customers spend {segment_stats.loc['💎 VIP Customers', 'avg_spend']/segment_stats.loc['📆 Occasional Buyers', 'avg_spend']:.1f}x more than occasional buyers

3. BEHAVIORAL PATTERNS:
   - VIP Customers prefer: {category_avg.loc['💎 VIP Customers'].idxmax()} (score: {category_avg.loc['💎 VIP Customers'].max():.1f})
   - Regular Shoppers prefer: {category_avg.loc['🔄 Regular Shoppers'].idxmax()} (score: {category_avg.loc['🔄 Regular Shoppers'].max():.1f})
   - Occasional Buyers prefer: {category_avg.loc['📆 Occasional Buyers'].idxmax()} (score: {category_avg.loc['📆 Occasional Buyers'].max():.1f})

{'='*60}
STRATEGIC RECOMMENDATIONS
{'='*60}

🎯 FOR VIP CUSTOMERS (High Value):
   • Implement exclusive loyalty program with tiered benefits
   • Offer early access to new products
   • Provide dedicated account manager
   • Send personalized premium offers
   • Expected ROI: High (retention focus)

🎯 FOR REGULAR SHOPPERS (Growth Potential):
   • Introduce subscription or membership program
   • Cross-sell based on category preferences
   • Bundle products from preferred categories
   • Implement referral bonuses
   • Expected ROI: Medium-High

🎯 FOR OCCASIONAL BUYERS (Re-engagement):
   • Send reactivation campaigns with special discounts
   • Create win-back offers after 30 days inactivity
   • Highlight new arrivals in their preferred categories
   • Survey to understand purchase barriers
   • Expected ROI: Medium

🎯 FOR NEW CUSTOMERS (Retention):
   • Welcome series with progressive discounts
   • Onboarding emails showcasing popular products
   • Request feedback after first month
   • Second purchase incentive (10-15% off)
   • Expected ROI: High (conversion focus)

{'='*60}
ACTION ITEMS
{'='*60}

✅ Immediate (Next 30 days):
   • Launch VIP loyalty program
   • Set up win-back email automation
   • Create welcome series for new customers

✅ Short-term (90 days):
   • Implement cross-selling recommendations
   • A/B test subscription offers
   • Develop segment-specific landing pages

✅ Long-term (6+ months):
   • Build predictive model for segment migration
   • Dynamic pricing by segment
   • Personalized product recommendations

{'='*60}
"""

print(insights)

# Save insights to file
with open("segment_insights.txt", "w") as f:
    f.write(insights)

print("✅ Insights saved to segment_insights.txt")

# ============================================
# 7. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 CUSTOMER SEGMENTATION COMPLETE!")
print("="*60)
print("\n📁 Files Generated:")
print("   1. customer_data.csv - Raw customer data (500 records)")
print("   2. segmentation_results.csv - Customers with segment labels")
print("   3. segment_analysis.html - Interactive 3D visualization")
print("   4. segment_distribution.png - Pie chart of segments")
print("   5. segment_profiles.png - Segment characteristics bar chart")
print("   6. category_preferences.png - Category preference heatmap")
print("   7. revenue_contribution.png - Revenue by segment chart")
print("   8. segment_insights.txt - Business insights & recommendations")
print("\n🚀 Next Steps:")
print("   1. Open segment_analysis.html in your browser")
print("   2. Review segment insights and recommendations")
print("   3. Export segmentation_results.csv for marketing tools")
print("   4. Implement targeted campaigns for each segment")
print("\n" + "="*60)
