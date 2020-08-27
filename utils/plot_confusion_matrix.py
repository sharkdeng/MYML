from sklearn.metrics import plot_confusion_matrix


def plot_confusion_matrix(cm, labels=range(6)):
    df_cm = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize=(10,  7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 13}) # font size
    plt.show()
