"""
Simplified Trend Analysis - Optimized for large datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def simplified_trend_analysis():
    """
    Execute streamlined trend analysis optimized for large datasets.
    
    Performs comprehensive YouTube trend analysis with efficient sampling,
    generates 12-panel visualization dashboard, builds simple prediction model,
    and provides strategic insights. Designed for fast execution on large datasets.
    
    Returns:
        tuple: (df_sample, model, scaler) containing:
            - df_sample: Preprocessed sample dataset with engineered features
            - model: Trained LinearRegression model for trend prediction
            - scaler: StandardScaler for feature normalization
    """
    print("🚀 SIMPLIFIED TREND ANALYSIS - OPTIMIZED VERSION")
    print("="*60)
    
    # Load and sample data for efficiency
    print("Loading and sampling data...")
    df = pd.read_csv('Dataset/videos.csv')
    
    # Sample 20,000 records for modeling to ensure reasonable execution time
    if len(df) > 20000:
        df_sample = df.sample(n=20000, random_state=42)
        print(f"Using sample of {len(df_sample)} records for modeling")
    else:
        df_sample = df.copy()
    
    # Basic preprocessing
    df_sample['publishedAt'] = pd.to_datetime(df_sample['publishedAt'])
    df_sample['year'] = df_sample['publishedAt'].dt.year
    df_sample['month'] = df_sample['publishedAt'].dt.month
    df_sample['hour'] = df_sample['publishedAt'].dt.hour
    df_sample['day_of_week'] = df_sample['publishedAt'].dt.dayofweek
    
    # Clean numeric columns
    numeric_cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
    for col in numeric_cols:
        df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce').fillna(0)
    
    # Create engagement score
    df_sample['engagement_score'] = (df_sample['likeCount'] + df_sample['commentCount']) / (df_sample['viewCount'] + 1)
    
    # Parse duration
    def parse_duration_simple(duration_str):
        """
        Parse YouTube duration format with simplified error handling.
        
        Converts ISO 8601 duration format to seconds with default fallback values
        for invalid or missing inputs. Optimized for performance over precision.
        
        Args:
            duration_str (str): YouTube duration string in PT format
        
        Returns:
            int: Duration in seconds, with defaults (60s for invalid, 1s minimum)
        """
        if pd.isna(duration_str) or duration_str == '':
            return 60  # Default 1 minute
        try:
            duration_str = str(duration_str).replace('PT', '')
            seconds = 0
            if 'H' in duration_str:
                hours = int(duration_str.split('H')[0])
                seconds += hours * 3600
                duration_str = duration_str.split('H')[1]
            if 'M' in duration_str:
                minutes = int(duration_str.split('M')[0])
                seconds += minutes * 60
                duration_str = duration_str.split('M')[1]
            if 'S' in duration_str:
                secs = int(duration_str.replace('S', ''))
                seconds += secs
            return max(seconds, 1)  # Minimum 1 second
        except:
            return 60
    
    df_sample['duration_seconds'] = df_sample['contentDuration'].apply(parse_duration_simple)
    
    print("\n📊 DATASET OVERVIEW")
    print("-" * 30)
    print(f"Total videos: {len(df):,}")
    print(f"Sampled videos: {len(df_sample):,}")
    print(f"Date range: {df_sample['publishedAt'].min()} to {df_sample['publishedAt'].max()}")
    print(f"Unique channels: {df_sample['channelId'].nunique():,}")
    print(f"Average views: {df_sample['viewCount'].mean():,.0f}")
    print(f"Average engagement: {df_sample['engagement_score'].mean():.4f}")
    
    # Create comprehensive visualizations
    print("\n🎨 CREATING VISUALIZATIONS")
    print("-" * 30)
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Views over time
    plt.subplot(3, 4, 1)
    monthly_views = df_sample.groupby(['year', 'month'])['viewCount'].mean()
    monthly_views.plot(kind='line', marker='o')
    plt.title('Average Views Over Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Upload frequency
    plt.subplot(3, 4, 2)
    upload_counts = df_sample.groupby(['year', 'month']).size()
    # upload_counts.plot(kind='bar', color='skyblue')
    # plt.title('Video Upload Frequency')
    # plt.xticks(rotation=45)
    # plt.grid(True, alpha=0.3)
    # Create proper date labels for x-axis
    date_labels = [f"{year}-{month:02d}" for year, month in upload_counts.index]
    plt.bar(range(len(upload_counts)), upload_counts.values, color='skyblue')
    plt.title('Video Upload Frequency')
    # Show only every 3rd label to prevent overcrowding
    step = max(1, len(date_labels) // 8)  # Show ~8 labels max
    plt.xticks(range(0, len(date_labels), step), 
            [date_labels[i] for i in range(0, len(date_labels), step)], 
            rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 3. Engagement distribution
    plt.subplot(3, 4, 3)
    engagement_filtered = df_sample['engagement_score'][df_sample['engagement_score'] < 0.5]
    plt.hist(engagement_filtered, bins=50, alpha=0.7, color='green')
    plt.title('Engagement Score Distribution')
    plt.xlabel('Engagement Score')
    plt.grid(True, alpha=0.3)
    
    # 4. Duration analysis
    plt.subplot(3, 4, 4)
    duration_bins = [0, 30, 60, 180, 600, float('inf')]
    duration_labels = ['<30s', '30s-1m', '1m-3m', '3m-10m', '>10m']
    df_sample['duration_cat'] = pd.cut(df_sample['duration_seconds'], bins=duration_bins, labels=duration_labels)
    duration_counts = df_sample['duration_cat'].value_counts()
    plt.pie(duration_counts.values, labels=duration_counts.index, autopct='%1.1f%%')
    plt.title('Duration Distribution')
    
    # 5. Posting patterns by day
    plt.subplot(3, 4, 5)
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = df_sample['day_of_week'].value_counts().sort_index()
    plt.bar(range(7), dow_counts.values, color='orange')
    plt.title('Posting by Day of Week')
    plt.xticks(range(7), dow_names)
    plt.grid(True, alpha=0.3)
    
    # 6. Posting patterns by hour
    plt.subplot(3, 4, 6)
    hour_counts = df_sample['hour'].value_counts().sort_index()
    plt.plot(hour_counts.index, hour_counts.values, marker='o', color='red')
    plt.title('Posting by Hour')
    plt.xlabel('Hour of Day')
    plt.grid(True, alpha=0.3)
    
    # 7. Views vs Duration
    plt.subplot(3, 4, 7)
    duration_perf = df_sample.groupby('duration_cat')['viewCount'].mean()
    plt.bar(range(len(duration_perf)), duration_perf.values, color='purple')
    plt.title('Views by Duration')
    plt.xticks(range(len(duration_perf)), duration_perf.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 8. Language distribution
    plt.subplot(3, 4, 8)
    lang_counts = df_sample['defaultLanguage'].value_counts().head(8)
    plt.barh(range(len(lang_counts)), lang_counts.values)
    plt.title('Top Languages')
    plt.yticks(range(len(lang_counts)), lang_counts.index)
    plt.grid(True, alpha=0.3)
    
    # 9. Yearly trends
    plt.subplot(3, 4, 9)
    yearly_stats = df_sample.groupby('year').agg({
        'viewCount': 'mean',
        'likeCount': 'mean',
        'commentCount': 'mean'
    })
    yearly_stats['viewCount'].plot(kind='line', marker='o', label='Views')
    plt.title('Yearly Performance Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Engagement vs Views scatter
    plt.subplot(3, 4, 10)
    sample_scatter = df_sample.sample(min(2000, len(df_sample)))
    plt.scatter(sample_scatter['viewCount'], sample_scatter['engagement_score'], alpha=0.6, s=10)
    plt.title('Views vs Engagement')
    plt.xlabel('Views (log scale)')
    plt.ylabel('Engagement Score')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 11. Top performers analysis
    plt.subplot(3, 4, 11)
    top_1000 = df_sample.nlargest(min(1000, len(df_sample)), 'viewCount')
    top_lang = top_1000['defaultLanguage'].value_counts().head(5)
    plt.bar(range(len(top_lang)), top_lang.values)
    plt.title('Top Performers by Language')
    plt.xticks(range(len(top_lang)), top_lang.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 12. Trend prediction simple model
    plt.subplot(3, 4, 12)
    
    # Simple linear trend model
    df_model = df_sample.copy()
    df_model['days_since_start'] = (df_model['publishedAt'] - df_model['publishedAt'].min()).dt.days
    
    # Features for simple model
    X = df_model[['days_since_start', 'month', 'hour', 'duration_seconds']].fillna(0)
    y = np.log1p(df_model['viewCount'])
    
    # Train simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    
    plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.6, s=10)
    plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], 
             [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--', lw=2)
    plt.title(f'Trend Prediction Model\nR² = {r2:.3f}')
    plt.xlabel('Actual Views')
    plt.ylabel('Predicted Views')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_trend_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 30)
    
    # Best performing characteristics
    top_performers = df_sample.nlargest(min(1000, len(df_sample)), 'viewCount')
    
    print(f"🎯 Top Performing Content:")
    print(f"   • Optimal duration: {top_performers['duration_seconds'].median():.0f} seconds")
    print(f"   • Best language: {top_performers['defaultLanguage'].mode().iloc[0] if not top_performers['defaultLanguage'].empty else 'N/A'}")
    print(f"   • Average engagement: {top_performers['engagement_score'].mean():.4f}")
    
    # Temporal patterns
    peak_hour = df_sample.groupby('hour')['viewCount'].mean().idxmax()
    peak_dow = df_sample.groupby('day_of_week')['viewCount'].mean().idxmax()
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print(f"\n⏰ Optimal Timing:")
    print(f"   • Best posting hour: {peak_hour}:00")
    print(f"   • Best day: {dow_names[peak_dow]}")
    
    # Content insights
    best_duration = df_sample.groupby('duration_cat')['viewCount'].mean().idxmax()
    print(f"\n📺 Content Strategy:")
    print(f"   • Best duration category: {best_duration}")
    print(f"   • Content variety: {df_sample['duration_cat'].nunique()} duration categories")
    
    # Model performance
    print(f"\n🤖 Model Performance:")
    print(f"   • Trend prediction R²: {r2:.3f}")
    print(f"   • Model can explain {r2*100:.1f}% of view variance")
    
    print(f"\n✅ Analysis complete! Generated 'comprehensive_trend_analysis.png'")
    
    return df_sample, model, scaler

if __name__ == "__main__":
    df_sample, model, scaler = simplified_trend_analysis()