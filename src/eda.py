import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

def load_data():
    """Load processed data"""
    return pd.read_csv(Path('data/processed/cleaned_data.csv'))

def plot_top_crops(df):
    """Top 10 crops by production"""
    top_crops = df.groupby('crop')['production'].sum().nlargest(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_crops.values, y=top_crops.index)
    plt.title("Top 10 Crops by Total Production")
    plt.tight_layout()
    plt.savefig('top_crops.png')
    plt.close()

def plot_yield_trends(df):
    """Yield trends over time"""
    top_5_crops = df.groupby('crop')['production'].sum().nlargest(5).index
    plt.figure(figsize=(12, 6))
    
    for crop in top_5_crops:
        crop_data = df[df['crop'] == crop]
        sns.lineplot(data=crop_data, x='year', y='yield', label=crop)
    
    plt.title("Yield Trends for Top 5 Crops")
    plt.legend()
    plt.tight_layout()
    plt.savefig('yield_trends.png')
    plt.close()

def analyze_data():
    """Run all EDA"""
    df = load_data()
    plot_top_crops(df)
    plot_yield_trends(df)
    
    # Correlation matrix
    corr = df[['area_harvested', 'yield', 'production']].corr()
    sns.heatmap(corr, annot=True)
    plt.savefig('correlation.png')
    plt.close()

if __name__ == "__main__":
    print("Running EDA...")
    analyze_data()
    print("EDA complete! Check generated plots.")