"""
Data Analysis and Trend Modeling for YouTube Video Dataset
Performs exploratory data analysis, trend identification, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the YouTube video dataset"""
    print("Loading dataset...")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Convert publishedAt to datetime
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Extract date components for trend analysis
    df['year'] = df['publishedAt'].dt.year
    df['month'] = df['publishedAt'].dt.month
    df['day_of_week'] = df['publishedAt'].dt.dayofweek
    df['hour'] = df['publishedAt'].dt.hour
    
    # Clean and convert numeric columns
    numeric_cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with 0 for engagement metrics
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Create engagement score
    df['engagement_score'] = (df['likeCount'] + df['commentCount']) / (df['viewCount'] + 1)
    
    # Parse content duration (convert PT format to seconds)
    df['duration_seconds'] = df['contentDuration'].apply(parse_duration)
    
    print("Data preprocessing completed.")
    return df

def parse_duration(duration_str):
    """Parse YouTube duration format (PT1M30S) to seconds"""
    if pd.isna(duration_str) or duration_str == '':
        return 0
    
    try:
        # Remove PT prefix
        duration_str = duration_str.replace('PT', '')
        
        seconds = 0
        # Parse hours
        if 'H' in duration_str:
            hours = int(duration_str.split('H')[0])
            seconds += hours * 3600
            duration_str = duration_str.split('H')[1]
        
        # Parse minutes
        if 'M' in duration_str:
            minutes = int(duration_str.split('M')[0])
            seconds += minutes * 60
            duration_str = duration_str.split('M')[1]
        
        # Parse seconds
        if 'S' in duration_str:
            secs = int(duration_str.replace('S', ''))
            seconds += secs
            
        return seconds
    except:
        return 0

def generate_basic_statistics(df):
    """Generate basic statistics and insights"""
    print("\n=== BASIC DATASET STATISTICS ===")
    print(f"Total videos: {len(df):,}")
    print(f"Date range: {df['publishedAt'].min()} to {df['publishedAt'].max()}")
    print(f"Unique channels: {df['channelId'].nunique():,}")
    
    print("\n=== ENGAGEMENT METRICS ===")
    engagement_stats = df[['viewCount', 'likeCount', 'commentCount', 'engagement_score']].describe()
    print(engagement_stats)
    
    print("\n=== CONTENT STATISTICS ===")
    print(f"Average video duration: {df['duration_seconds'].mean():.1f} seconds")
    print(f"Most common language: {df['defaultLanguage'].value_counts().head(1).index[0] if not df['defaultLanguage'].isna().all() else 'N/A'}")
    
    return engagement_stats

def analyze_trends_over_time(df):
    """Analyze trends over time"""
    print("\n=== TEMPORAL TREND ANALYSIS ===")
    
    # Monthly trends
    monthly_trends = df.groupby(['year', 'month']).agg({
        'viewCount': 'mean',
        'likeCount': 'mean', 
        'commentCount': 'mean',
        'engagement_score': 'mean',
        'videoId': 'count'
    }).round(2)
    
    monthly_trends.columns = ['Avg_Views', 'Avg_Likes', 'Avg_Comments', 'Avg_Engagement', 'Video_Count']
    
    print("Monthly aggregated metrics:")
    print(monthly_trends.tail(10))
    
    return monthly_trends

def create_visualizations(df, monthly_trends):
    """Create comprehensive visualizations"""
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Trend over time - Video uploads
    plt.subplot(3, 3, 1)
    video_counts = df.groupby('year')['videoId'].count()
    plt.plot(video_counts.index, video_counts.values, marker='o', linewidth=2)
    plt.title('Video Uploads Trend Over Years', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Videos')
    plt.grid(True, alpha=0.3)
    
    # 2. Average views over time
    plt.subplot(3, 3, 2)
    yearly_views = df.groupby('year')['viewCount'].mean()
    plt.plot(yearly_views.index, yearly_views.values, marker='s', color='orange', linewidth=2)
    plt.title('Average Views Trend Over Years', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Average Views')
    plt.grid(True, alpha=0.3)
    
    # 3. Engagement score distribution
    plt.subplot(3, 3, 3)
    plt.hist(df['engagement_score'][df['engagement_score'] < 1], bins=50, alpha=0.7, color='green')
    plt.title('Engagement Score Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Engagement Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 4. Duration analysis
    plt.subplot(3, 3, 4)
    duration_bins = [0, 30, 60, 180, 600, float('inf')]
    duration_labels = ['<30s', '30s-1m', '1m-3m', '3m-10m', '>10m']
    df['duration_category'] = pd.cut(df['duration_seconds'], bins=duration_bins, labels=duration_labels)
    duration_counts = df['duration_category'].value_counts()
    plt.pie(duration_counts.values, labels=duration_counts.index, autopct='%1.1f%%')
    plt.title('Video Duration Distribution', fontsize=12, fontweight='bold')
    
    # 5. Day of week posting pattern
    plt.subplot(3, 3, 5)
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = df['day_of_week'].value_counts().sort_index()
    plt.bar(range(7), dow_counts.values, color='skyblue')
    plt.title('Posting Pattern by Day of Week', fontsize=12, fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Videos')
    plt.xticks(range(7), dow_names)
    plt.grid(True, alpha=0.3)
    
    # 6. Hour of day posting pattern
    plt.subplot(3, 3, 6)
    hour_counts = df['hour'].value_counts().sort_index()
    plt.plot(hour_counts.index, hour_counts.values, marker='o', color='red')
    plt.title('Posting Pattern by Hour', fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Videos')
    plt.grid(True, alpha=0.3)
    
    # 7. Views vs Engagement scatter
    plt.subplot(3, 3, 7)
    sample_data = df.sample(min(5000, len(df)))  # Sample for performance
    plt.scatter(sample_data['viewCount'], sample_data['engagement_score'], alpha=0.6, s=20)
    plt.title('Views vs Engagement Score', fontsize=12, fontweight='bold')
    plt.xlabel('View Count')
    plt.ylabel('Engagement Score')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 8. Language distribution (top 10)
    plt.subplot(3, 3, 8)
    lang_counts = df['defaultLanguage'].value_counts().head(10)
    plt.barh(range(len(lang_counts)), lang_counts.values)
    plt.title('Top 10 Content Languages', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Videos')
    plt.yticks(range(len(lang_counts)), lang_counts.index)
    plt.grid(True, alpha=0.3)
    
    # 9. Monthly trend line
    plt.subplot(3, 3, 9)
    monthly_data = monthly_trends.reset_index()
    monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
    plt.plot(monthly_data['date'], monthly_data['Avg_Views'], marker='o', label='Avg Views')
    plt.plot(monthly_data['date'], monthly_data['Avg_Likes'] * 10, marker='s', label='Avg Likes (x10)')
    plt.title('Monthly Trends', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Metrics')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'trend_analysis_dashboard.png'")

def main():
    """Main analysis function"""
    print("Starting YouTube Video Trend Analysis...")
    
    # Load and preprocess data
    df = load_and_preprocess_data('Dataset/videos.csv')
    
    # Generate basic statistics
    stats = generate_basic_statistics(df)
    
    # Analyze trends over time
    monthly_trends = analyze_trends_over_time(df)
    
    # Create visualizations
    create_visualizations(df, monthly_trends)
    
    print("\nAnalysis completed successfully!")
    return df, monthly_trends, stats

if __name__ == "__main__":
    df, monthly_trends, stats = main()