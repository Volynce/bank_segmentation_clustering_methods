import pandas as pd

def load_and_clean_data(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path, sep='\t', encoding='cp1252', decimal=',')
    
    # Очистка данных
    df.dropna(inplace=True)
    df = df[df['Age of client'] >= 18]
    df = df.sample(n=10000, random_state=1)  # Сэмплирование данных

    # Удаление целевой переменной для кластеризации
    df_dr = df.drop(columns=['TARGET (take a credit)'])
    
    return df, df_dr