"""main logic of the project application"""
import functions as fc

if __name__ == "__main__":
    print "Connection"
    fc.set_connect()
    print "Done!"

    print "Read tables"
    anime, anime_list = fc.read_tables()
    print "Done!"

    print "Preprocess data"
    anime_df_filtered = fc.preprocess_data(anime, anime_list)
    print "Done!"

    print "Calculate User-Item matrix"
    transformed = fc.create_matrix(anime_df_filtered)
    train_df, test_df = transformed.randomSplit([0.7, 0.3])
    print "Done!"

    print "Get true values"
    items_for_user_true = fc.calculate_true_values(test_df)
    print "Done!"

    # baseline solution
    print "Run baseline solution"
    baseline_pred = fc.baseline(train_df)
    print "Done!"

    print "Evaluate Baseline solution"
    fc.evaluate("BASELINE", baseline_pred, items_for_user_true)
    print "Done!"

    # ALS solution
    print "Run first model (ALS)"
    als = fc.first_model(train_df)
    print "Done!"

    print "Make prediction"
    als_pred = fc.make_pred_first_model(als, train_df, transformed)
    print "Done!"

    print "Evaluate"
    fc.evaluate("ALS", als_pred, items_for_user_true)
    print "Done!"

    # ALS GridSearch
    print "Run ALS GridSearch"
    als_grid = fc.grid_search_first_model(train_df)
    print "Done!"

    print "Make prediction of best model ALS"
    als_pred = fc.make_pred_first_model(als, train_df, transformed)
    print "Done!"

    print "Evaluation of the best ALS model"
    fc.evaluate("GRIDSEARCH ALS", als_pred, items_for_user_true)
    print "Done!"

    print "Close connect"
    fc.close_connect()
    print "Done!"