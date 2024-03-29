from itertools import product
import numpy as np
import json
import os

def save_to_json(exp_histories: list[dict], exp_reports: list[dict], 
                 history_path: str, report_path: str):
    """Save results from grid_search to json"""
    with open(history_path, 'w') as fp:
        json.dump(exp_histories, fp)

    with open(report_path, 'w') as fp:
        json.dump(exp_reports, fp)


def load_json(history_path: str, report_path: str) -> tuple[dict, dict]:
    """Save results from grid_search to json"""
    with open(history_path, 'r') as fp:
        exp_history = json.load(fp)

    with open(report_path, 'r') as fp:
        exp_report = json.load(fp)

    return exp_history, exp_report


def run_train_test(X_train, labels, X_test, Y_test, build_func, 
                   evaluate_func, epochs=25, batch_size=8, validation_split=0.0, 
                   **parameters):
    """
    Train and test the model.
    """
    model = build_func(**parameters)
    history = model.fit(X_train, labels, validation_split=validation_split, batch_size=batch_size, epochs=epochs)
    report = evaluate_func(model, X_test, Y_test)
    return history.history, report


def repeat_train_test(X_train, labels, X_test, Y_test, build_func, 
                      evaluate_func, repeat=10, epochs=25, batch_size=8, 
                      validation_split=0.0, **build_kwargs):
    """
    Train and test the model "repeat" amount of times.
    """
    run_history = []
    run_report = []
    for j in range(repeat):
        print(f'Run #{j + 1}')
        history, report = run_train_test(X_train, labels, X_test, Y_test,
                                         build_func, evaluate_func, epochs=epochs, 
                                         batch_size=batch_size, validation_split=validation_split, 
                                         **build_kwargs)
        run_history.append(history)
        run_report.append(report)

    return run_history, run_report


def grid_search(X_train, labels, X_test, Y_test, build_func, 
                evaluate_func, repeat=10, epochs=25, batch_size=8, 
                validation_split=0.0, report_path="", history_path="", **build_kwargs):
    """
    Train and test the model for each combination of build parameters "repeat" amount of times. 
    """
    key_args = build_kwargs.keys()
    exp_histories = [] #histories of each experiment
    exp_reports = [] #reports of each experiment

    if report_path != "" and history_path != "" and os.path.exists(report_path):
        exp_histories, exp_reports = load_json(history_path, report_path)

    if len(key_args) == 0:
        run_history, run_report = repeat_train_test(X_train, labels, X_test, Y_test,
                                                    build_func, evaluate_func, epochs=epochs, 
                                                    repeat=repeat, batch_size=batch_size, 
                                                    validation_split=validation_split)
        exp_histories.append({"args": {}, "histories": run_history})
        exp_reports.append({"args": {}, "reports": run_report})

        save_to_json(exp_histories, exp_reports, history_path, report_path)

    else:
        args_combinations = product(*build_kwargs.values())
        done = set()
        if len(exp_reports) != 0:
            for exp in exp_reports:
                args = tuple(exp['args'].values())
                done.add(args)
            print("Finished experiments:", done)
            args_combinations = list(set(args_combinations) - done)
        

        for args in args_combinations:
            kwargs = {k:v for k, v in zip(key_args, args)}
            print(f'Experiment with parameters: {kwargs}')

            run_history, run_report = repeat_train_test(X_train, labels, X_test, Y_test,
                                                        build_func, evaluate_func, repeat=repeat, 
                                                        epochs=epochs, batch_size=batch_size, 
                                                        validation_split=validation_split, **kwargs)

            exp_histories.append({"args": kwargs, "histories": run_history})
            exp_reports.append({"args": kwargs, "reports": run_report})

            save_to_json(exp_histories, exp_reports, history_path, report_path)

    return exp_histories, exp_reports


def show_results(exp_histories, exp_reports, K):
    """
    Print the results of the grid search
    """
    for histories, reports in zip(exp_histories, exp_reports):

        print("==================================================")
        print(f'parameters = {reports["args"]}')

        f1_scores = [[], [], []]
        accuracies = []
        for i, report in enumerate(reports["reports"]):
            for k in range(K):
                if report[str(k)]['f1-score'] == 0:
                    print(f'Warning in run #{i}. Class {k} has an f1_score of 0.0')
                f1_scores[k].append(report[str(k)]['f1-score'])


            accuracies.append(report['accuracy'])

        mean_f1_scores = [np.mean(f1_scores[k]) for k in range(K)]
        std_f1_scores = [np.std(f1_scores[k]) for k in range(K)]

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)


        print(f'\tmean f1 scores: {mean_f1_scores}')
        print(f'\tstd f1 scores: {std_f1_scores}')
        print(f'\tmean accuracy: {mean_accuracy}')
        print(f'\tstd accuracy: {std_accuracy}')
