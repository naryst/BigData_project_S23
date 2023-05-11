import functions as fc

if __name__ == "__main__":
    fc.set_connect()

    anime, anime_list = fc.read_tables()

    anime_df_filtered = fc.preprocess_data(anime, anime_list)

    transformed = fc.create_matrix(anime_df_filtered)
    train_df, test_df = transformed.randomSplit([0.7, 0.3])
    items_for_user_true = fc.calculate_true_values(test_df)

    # baseline solution
    baseline_pred = fc.baseline(train_df)
    fc.evaluate("BASELINE", baseline_pred, items_for_user_true)

    # ALS solution
    als = fc.first_model(train_df)
    als_pred = fc.make_pred_first_model(als, train_df, transformed)
    fc.evaluate("ALS", als_pred, items_for_user_true)

    # ALS GridSearch
    als_grid = fc.grid_search_first_model(train_df)
    als_pred = fc.make_pred_first_model(als, train_df, transformed)
    fc.evaluate("GRIDSEARCH ALS", als_pred, items_for_user_true)

    fc.close_connect()