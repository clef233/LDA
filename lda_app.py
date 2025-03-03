import streamlit as st
import tomotopy as tp
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora
import pyLDAvis
import pandas as pd
import jieba
import jieba.analyse
from cycler import cycler
import os
import streamlit.components.v1 as components
from tqdm import tqdm
import matplotlib.font_manager as fm

# 添加字体文件路径
font_path = 'SimHei.ttf'  
font_name = fm.FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示


@st.cache_data
def load_data(data_file, stopwords_file):
    """加载数据和停用词"""
    data_lda = pd.read_csv(data_file)
    data_lda['time'] = pd.to_datetime(data_lda['time'])
    data_lda['year'] = data_lda['time'].dt.year
    data_lda['text'] = data_lda['text'].fillna("").astype(str)

    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f.readlines()])
    return data_lda, stopwords

@st.cache_resource
def train_lda_model(data_lda, stopwords, num_topics):
    """训练 LDA 模型"""
    # 添加自定义词汇和词频
    jieba.add_word('乡村振兴', freq=50)

    # 分词并去除停用词
    texts = []
    for text in data_lda['text']:
        words = jieba.cut(text)
        clean_words = [word for word in words if word not in stopwords and word.strip()]
        texts.append(clean_words)

    # 创建词典和语料库
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 转换为 tomotopy 所需的格式
    corpus_tomotopy = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]

    # 创建 tomotopy LDA 模型实例
    lda_model = tp.LDAModel(k=num_topics, min_cf=20, seed=42)

    # 添加文档到模型中
    for doc in tqdm(corpus_tomotopy, desc="Adding documents to LDA model"):
        lda_model.add_doc(doc)

    # 训练 LDA 模型
    lda_model.train(0)
    st.write(f'Num docs: {len(lda_model.docs)}, Vocab size: {lda_model.num_vocabs}, Num words: {lda_model.num_words}')
    st.write(f'Removed top words: {lda_model.removed_top_words}')

    for i in range(1000):
        lda_model.train(1)

    return lda_model, dictionary

def display_topic_words(lda_model, topic_names):
    """打印主题词"""
    st.write("<h4>主题词</h4>", unsafe_allow_html=True)
    for k in range(lda_model.k):
        topic_words = lda_model.get_topic_words(k, top_n=10)
        st.write(f'{topic_names[k]}: {topic_words}')
    st.write("")

def calculate_topic_strengths(lda_model, data_lda):
    """计算每年每个主题的强度"""
    # 使用 data_lda 的索引创建 doc_topic_dist
    doc_topic_dist = {}
    for index, doc in zip(data_lda.index, lda_model.docs):
        doc_topic_dist[index] = doc.get_topic_dist()

    topic_strengths = {}
    for year, group in data_lda.groupby('year'):
        topic_strength = {}
        if len(group) == 0:
            st.write(f"警告：年份 {year} 没有数据。")
            topic_strengths[year] = topic_strength
            continue
        for k in range(lda_model.k):
            sum_topic_dist = 0
            for idx in group.index:
                try:
                    sum_topic_dist += doc_topic_dist[idx][k]
                except KeyError:
                    # 忽略 KeyError 警告
                    pass
            topic_strength[k] = sum_topic_dist / len(group)
        topic_strengths[year] = topic_strength
    return topic_strengths

def plot_topic_evolution(topic_strengths, topic_names, lda_model):
    """绘制主题强度演化图"""
    extended_color_scheme = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
        '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    extended_markers = ['o', '^', 's', 'D', 'v', 'p', '*', 'X', 'P', 'H', '8', 'd', '>', '<', 'h', '1', '2', '3', '4', '+']

    plt.figure(figsize=(12, 8), dpi=300)
    plt.rc('axes', prop_cycle=(cycler('color', extended_color_scheme) + cycler('marker', extended_markers)))
    plt.rc('lines', linewidth=2, markersize=8)

    for k in range(lda_model.k):
        topic_strength_over_years = [topic_strengths[year][k] for year in sorted(topic_strengths)]
        plt.plot(sorted(topic_strengths), topic_strength_over_years, label=topic_names[k])  # 使用中文名称
        plt.text(max(topic_strengths), topic_strength_over_years[-1], f'{topic_strength_over_years[-1]:.2f}', fontsize=10, va='center')

    plt.xlabel('年份')  # 修改 x 轴标签
    plt.ylabel('主题强度')  # 修改 y 轴标签
    plt.title('主题强度随年份的演化')  # 修改标题
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'SimHei'})  # 添加中文字体设置
    plt.tight_layout()
    st.pyplot(plt)

def display_pyldavis(lda_model, dictionary, data_lda):
    """显示 pyLDAvis 可视化"""
    topic_term_dists = np.stack([lda_model.get_topic_word_dist(k) for k in range(lda_model.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in lda_model.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in lda_model.docs])
    vocab = list(lda_model.used_vocabs)
    term_frequency = lda_model.used_vocab_freq

    prepared_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency,
        start_index=0,  # tomotopy话题id从0开始
        sort_topics=False,  # 注意:否则pyLDAvis与tomotopy内的话题无法一一对应
        n_jobs=1  # 避免使用并行处理
    )

    # 将 pyLDAvis 输出保存为 HTML 文件
    html_file = 'lda_visualization.html'
    pyLDAvis.save_html(prepared_data, html_file)

    # 在 Streamlit 中显示 HTML 文件
    with open(html_file, 'r', encoding='utf-8') as f:
        html_string = f.read()
    components.html(html_string, width=1300, height=800, scrolling=True)

def main():
    st.title("LDA 主题模型分析")

    # 文件上传
    data_file = st.file_uploader("上传数据 CSV 文件", type=["csv"])
    stopwords_file = st.file_uploader("上传停用词 TXT 文件", type=["txt"])

    if data_file is not None and stopwords_file is not None:
        # 保存上传的文件到临时目录
        with open(os.path.join("tempDir",data_file.name),"wb") as f:
          f.write(data_file.getbuffer())
        with open(os.path.join("tempDir",stopwords_file.name),"wb") as f:
          f.write(stopwords_file.getbuffer())
        data_file_path = os.path.join("tempDir",data_file.name)
        stopwords_file_path = os.path.join("tempDir",stopwords_file.name)

        num_topics = st.slider("选择主题数量", min_value=2, max_value=20, value=8)

        data_lda, stopwords = load_data(data_file_path, stopwords_file_path)
        lda_model, dictionary = train_lda_model(data_lda, stopwords, num_topics)

        # 打印主题，并为每个主题分配一个中文名称
        topic_names = {
            i: st.text_input(f"主题 {i+1} 名称", f"主题 {i+1}") for i in range(num_topics)
        }

        display_topic_words(lda_model, topic_names)

        # 计算每年每个主题的强度
        topic_strengths = calculate_topic_strengths(lda_model, data_lda)

        # 绘制主题强度随年份的演化
        plot_topic_evolution(topic_strengths, topic_names, lda_model)

        # 显示 pyLDAvis 可视化
        display_pyldavis(lda_model, dictionary, data_lda)

# 创建临时目录
if not os.path.exists("tempDir"):
    os.makedirs("tempDir")
if __name__ == "__main__":
    main()
