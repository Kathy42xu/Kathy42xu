import streamlit as st
import os
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Market Data Analysis Dashboard", layout="wide")

st.title("Market Data Analysis Dashboard")
st.markdown("""
This web app analyzes market data to deliver actionable insights.
It helps businesses understand their market position, competition, and growth potential.
""")

task_option = st.sidebar.selectbox(
    "Select Task:",
    ("Task 1: Customer Sentiment Analysis", "Task 2: Market Position Analysis / Suggested Actions")
)

def get_image(image_name):
    return os.path.join("www", image_name)

if task_option == "Task 1: Customer Sentiment Analysis":
    st.header("Task 1: Customer Sentiment Analysis")
    
    st.markdown("""
    **Sentiment Breakdown (in percentages):**
    ```
    sentiment
    positive    85.243272
    neutral     10.051684
    negative     4.705044
    ```
    
    **Key Phrases and Themes:**
    ```
          Key Phrase      Frequency
    0            love            385
    1       beautiful            211
    2   love curtains            131
    3             amp            110
    4            nice              79
    5      look great             76
    6              br             67
    7   great quality             66
    8    good quality             60
    9         perfect             58
    10          thank             46
    11          happy             43
    12   disappointed             43
    13        awesome             42
    14         pretty             42
    15        however             37
    16            see             35
    17      well made             35
    18          thick             33
    19          great             32
    ```
    
    **Key Findings:**
    
    The key phrase analysis reveals overwhelmingly positive customer sentiment with words like "love" and "beautiful" appearing most frequently, indicating strong emotional approval of the product. Phrases such as "love curtains", "great quality", "good quality", and "perfect" suggest that quality and design are highly valued. However, the presence of "disappointed" among mostly positive terms indicates isolated concerns that might warrant further investigation.
    
    **Identified Customer Pain Points (Topics):**
    - **Topic 1:** 0.161*"like" + 0.115*"look" + 0.106*"picture" + 0.094*"nothing" + 0.074*"curtain" + 0.026*"color"
    - **Topic 2:** 0.091*"br" + 0.066*"curtain" + 0.066*"two" + 0.057*"panel" + 0.036*"room" + 0.029*"one"
    - **Topic 3:** 0.073*"look" + 0.068*"bad" + 0.045*"disappointed" + 0.042*"quality" + 0.042*"picture" + 0.038*"one"
    - **Topic 4:** 0.105*"curtain" + 0.048*"quality" + 0.043*"material" + 0.032*"disappointed" + 0.031*"return" + 0.028*"cheap"
    - **Topic 5:** 0.097*"wrong" + 0.059*"back" + 0.055*"curtain" + 0.051*"size" + 0.043*"sent" + 0.042*"fabric"
    - **Topic 6:** 0.067*"disappointed" + 0.061*"like" + 0.054*"picture" + 0.048*"curtain" + 0.046*"poor" + 0.043*"look"
    - **Topic 7:** 0.051*"curtain" + 0.044*"really" + 0.034*"back" + 0.033*"thought" + 0.033*"send" + 0.028*"complaint"
    - **Topic 8:** 0.076*"color" + 0.049*"fabric" + 0.049*"block" + 0.043*"picture" + 0.042*"cheap" + 0.041*"light"
    - **Topic 9:** 0.062*"look" + 0.055*"br" + 0.052*"curtain" + 0.032*"blurry" + 0.027*"never" + 0.027*"mistake"
    - **Topic 10:** 0.048*"curtain" + 0.046*"wanted" + 0.046*"ugly" + 0.045*"bad" + 0.044*"product" + 0.039*"looked"
    """)
    
    st.image(get_image("1.png"), caption="Customer Sentiment Analysis", use_container_width=True)
    
    st.markdown("""
    **Short Write-up for Our Findings:**
    
    Our analysis of the negative reviews using topic modeling revealed several recurring pain points. In many topics, customers repeatedly mention issues with the aesthetic and material quality of the curtains. For example, topics 1, 3, 6, and 10 include terms like "look," "picture," "bad," "disappointed," and "ugly," suggesting dissatisfaction with visual appeal and overall quality. Other topics (e.g., Topics 4 and 8) emphasize problems with material and fabric quality, indicating that the build and finish may not meet expectations. Additionally, some topics hint at issues related to product sizing or incorrect orders, while others suggest customers feel compelled to complain or return the product.
    """)
    
else:
    st.header("Task 2: Market Position Analysis / Suggested Actions")
    st.markdown("This section analyzes market data to determine product pricing advantage and offers strategic recommendations.")
    
    @st.cache_data
    def load_market_data():
        df_market = pd.read_csv('/Users/xuhuirong/Desktop/Product.csv')
        df_market['Promotion'] = df_market['Promotion'].map({'Yes': 1, 'No': 0})
        df_market['Price_Ratio'] = df_market['Price'] / df_market["Competitor's Price"]
        return df_market
    df_market = load_market_data()
    st.write("### Market Data Overview", df_market.head())
    
    features = df_market[['Promotion', 'Foot Traffic', 'Product Position', 'Consumer Demographics', 'Product Category', 'Seasonal']]
    features_encoded = pd.get_dummies(features, columns=['Product Position', 'Foot Traffic', 'Consumer Demographics', 'Product Category', 'Seasonal'])
    target = df_market['Price_Ratio']
    st.write("### Data Overview")
    st.image(get_image("2.png"), caption="Data overview", use_container_width=True)
    st.write("### Correlation barplot")
    st.image(get_image("3.png"), caption="Correlation barplot", use_container_width=True)
    st.write("### Correlation Matrix")
    df_corr = features_encoded.copy()
    df_corr['Price_Ratio'] = target
    fig_corr = px.imshow(df_corr.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Random Forest Test MSE:** {mse:.2f}")
    
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    st.write("### Feature Importance")
    fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance',
                            title="Feature Importance from Random Forest Model")
    st.plotly_chart(fig_importance)
    
    fig1 = px.box(df_market, x='Promotion', y='Price_Ratio', title="Price_Ratio by Promotion (0: No, 1: Yes)")
    st.plotly_chart(fig1)
    
    fig2 = px.box(df_market, x='Product Position', y='Price_Ratio', title="Price_Ratio by Product Position")
    st.plotly_chart(fig2)
    
    st.write("## Suggested Actions")
    st.markdown("""
    ## 3. Suggested Actions

### 3.1 Improvements: Product or Service Enhancements

1. **Revise Promotional Strategies:**  
   - **Redesign Promotions:** Instead of deep discounts that lower the Price Ratio, explore value-added offers such as bundling or loyalty rewards.
   - **Optimize Timing:** Use A/B testing to determine which promotional styles yield higher Price Ratios and adjust campaigns accordingly.

2. **Optimize Shelf Placement & Visibility:**  
   - **Prime Locations:** Secure high-traffic areas (e.g., aisle and front-of-store) to maximize product visibility.
   - **Enhanced Merchandising:** Invest in improved in-store signage and visual displays to reinforce product value and justify a premium price.

3. **Enhance Product Presentation:**  
   - **Packaging & Quality:** Upgrade product packaging and overall presentation to bolster perceived quality, helping to sustain a higher Price Ratio even during promotions.

---

### 3.2 Strategic Advice: Market Positioning Strategies

1. **Dynamic Pricing Strategies:**  
   - **Real-Time Adjustments:** Implement dynamic pricing that adjusts based on competitor pricing and current market conditions, ensuring the product maintains a premium position.
   - **Segment-Based Pricing:** Leverage consumer demographic data to tailor pricing for different market segments.

2. **Targeted Marketing Initiatives:**  
   - **Customized Messaging:** Develop marketing messages and promotional strategies tailored to specific consumer segments that are more likely to respond to premium pricing.
   - **Cross-Channel Integration:** Use insights from Foot Traffic and Seasonal data to coordinate online and in-store promotions effectively.

3. **Balanced Product Portfolio:**  
   - **Diverse Offerings:** Maintain a mix of products with varying margins; focus on promoting those that can be positioned at premium spots in the store.

---

### 3.3 Actionable Insights: Leveraging Market Trends

1. **Develop a Real-Time Dashboard:**  
   - **Monitor Key Metrics:** Create a dashboard to continuously track Price Ratio, Promotion effectiveness, Foot Traffic, and seasonal trends, enabling rapid response to market shifts.

2. **Experiment with Promotion Formats:**  
   - **A/B Testing:** Conduct experiments with different promotion types (e.g., value-add vs. discount) to identify the most effective methods for sustaining a high Price Ratio.

3. **Continuous Model Refinement:**  
   - **Data-Driven Decisions:** Regularly update your predictive models with new data to reassess feature importance and adapt your strategies based on the latest trends.

---

## 4. Conclusion

Our analysis shows that while promotions generally lower the Price Ratio, they remain a critical driver. By refining promotional strategies to add value, optimizing product placement, and leveraging seasonal and demographic insights, businesses can enhance their competitive positioning. Implementing dynamic pricing and continuously monitoring key metrics will ensure that strategic adjustments can be made swiftly to capitalize on market trends and sustain a premium pricing advantage.

    """)

st.write("## End of Dashboard")

