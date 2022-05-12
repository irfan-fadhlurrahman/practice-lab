import pandas as pd
import inflection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = ['#99d594', '#D53E4F', '#FC8D59']

def single_countplot(df, var):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['#99d594', '#D53E4F', '#FC8D59']
    sns.countplot(data=df, x=var, ax=ax, color=colors[0])
    feat_counts = df[var].value_counts().sort_index().values
    for n, count in enumerate(feat_counts):
        ax.annotate(
              f"{(100* count/np.sum(feat_counts)):.1f}%",
              xy=(n, count + 200),
              color='#4a4a4a', fontsize=14, alpha=1,
              va = 'center', ha='center',
        )
        for location in ['top', 'right']:
            ax.spines[location].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"{inflection.titleize(var)}", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        #ax.set_ylim(0, 350001, 50000)

    plt.xticks(alpha=1, fontsize=14)
    plt.yticks(alpha=1, fontsize=14)
    plt.grid(axis='y', alpha=0.5)

def cat_distribution(full_train, fig, ax, n_row=0, vars=None):
    for i, feat in enumerate(vars): 
      data = full_train[feat].value_counts().sort_index()
      sns.barplot(x=data.index, y=data.values, ax=ax[n_row, i], color=colors[0])
      for n, count in enumerate(data.values):
          ax[n_row, i].annotate(
              f"{(100* count/np.sum(data.values)):.1f}%", 
              xy=(n, count + 300), 
              color='#4a4a4a', fontsize=14,
              va = 'center', ha='center', 
          )
      for location in ['top', 'right']:
          ax[n_row, i].spines[location].set_visible(False)
      ax[n_row, i].set_title(f'{feat}\n', loc='center', pad=10, fontsize=14, fontweight='bold')
      ax[n_row, i].grid(axis='y', alpha=0.3)
      ax[n_row, i].set_axisbelow(True)
      ax[n_row, i].set_xlabel("")
      ax[n_row, i].set_ylim(0, 8001, 2000)
    
    plt.xticks(alpha=1, fontsize=14)
    plt.yticks(alpha=1, fontsize=14)

def box_bivariate(full_train, fig, ax, n_row=0, vars=None):
    for i, feat in enumerate(vars): 
        sns.boxplot(data=full_train, x=feat, ax=ax[i], palette=colors, width=0.3)
        total = full_train[feat].count()
        for p in ax[i].patches:
            ax[i].annotate(
                f"{p.get_width() / total * 100 :.1f}%", 
                xy=(p.get_width(), p.get_y()+p.get_height()/2),
                xytext=(5, 0), 
                textcoords='offset points', 
                ha="left", va="center"
            )
        for location in ['top', 'right', 'left']:
            ax[i].spines[location].set_visible(False)
        ax[i].set_title(f'\n{feat}\n', loc='center', pad=10, fontsize=14, fontweight='bold')
        ax[i].grid(axis='x', alpha=0.3)
        ax[i].set_axisbelow(True)
        ax[i].set_ylabel("")
        ax[i].set_xlabel("", fontsize=14)


    





