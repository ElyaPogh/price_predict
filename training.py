from script import DBConnect, Modeling, Preprocess

if __name__ == '__main__':

    conn = DBConnect('resources/credentials_view.json', "candidate-testing.house_data", "london_house_prices")
    conn.save('assets/data_table.csv')
    conn.load_dataframe()
    df = conn.get_dataframe()
    prep = Preprocess(df)
    pp_df = prep.feature_engineering()
    modl = Modeling()
    X_train, X_test, y_train, y_test = modl.get_data_prepared(pp_df)
    modl.get_model()
    model_n, summary = modl.train_model()
    modl.save_model('model/price_prediction_new.h5')
    modl.save_config('model/config.ini')
    model_pr = modl.load_model('model/price_prediction_new.h5')
