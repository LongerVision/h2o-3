from __future__ import print_function

import os
import sys

sys.path.insert(1, os.path.join("..", "..", ".."))
import matplotlib
matplotlib.use("Agg")  # remove warning from python2 (missing TKinter)
import h2o
import matplotlib.pyplot
from tests import pyunit_utils
from h2o.estimators import *


def test_display_mode():
    train = h2o.upload_file(pyunit_utils.locate("smalldata/titanic/titanic_expanded.csv"))
    y = "fare"

    gbm = H2OGradientBoostingEstimator(seed=1234, model_id="my_awesome_model")
    gbm.train(y=y, training_frame=train)

    assert isinstance(gbm.ice_plot(train, 'title').figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'title', show_pdp=True).figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'title', show_pdp=False).figure(), matplotlib.pyplot.Figure)

    assert isinstance(gbm.ice_plot(train, 'age').figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'age', show_pdp=True).figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'age', show_pdp=False).figure(), matplotlib.pyplot.Figure)
    matplotlib.pyplot.close("all")

def test_grouping_variable():
    paths = ["smalldata/titanic/titanic_expanded.csv", "smalldata/logreg/prostate.csv", "smalldata/iris/iris2.csv"]
    ys = ["fare", "CAPSULE", "response"]
    names_to_extract = ["name", None, None]
    targets = [None, None, "setosa"]
    cols_as_factors = [None, ["GLEASON", "CAPSULE"], None]
    grouping_variable = ["cabin_type", "GLEASON", "response"]

    for i in range(len(paths)):
        train = h2o.upload_file(pyunit_utils.locate(paths[i]))
        if cols_as_factors[i] is not None:
            for col in cols_as_factors[i]:
                train[col] = train[col].asfactor()
        gbm = H2OGradientBoostingEstimator(seed=1234, model_id="my_awesome_model_py" + str(i))
        gbm.train(y=ys[i], training_frame=train)

        cols_to_test = _get_cols_to_test(train, ys[i])
        if names_to_extract[i] is not None:
            if names_to_extract[i] in cols_to_test: cols_to_test.remove(names_to_extract[i])

        _assert_list_of_plots_was_produced(cols_to_test, gbm, train, targets[i], grouping_variable[i])


def _get_cols_to_test(train, y):
    cols_to_test = []
    for col, typ in train.types.items():
        for ctt in cols_to_test:
            if typ == train.types[ctt] or col == y:
                break
        else:
            cols_to_test.append(col)
    return cols_to_test

def _assert_list_of_plots_was_produced(cols_to_test, model, train, target, grouping_variable):
    for col in cols_to_test:
        if target is None:
            save_plot_path = model.training_model_metrics()["model_category"] + "_" + col + ".png"
            ice_plot_result = model.ice_plot(train, col, grouping_variable=grouping_variable)
        else:
            save_plot_path = model.training_model_metrics()["model_category"] + "_" + target + "_" + col + ".png"
            ice_plot_result = model.ice_plot(train, col, target="setosa", grouping_variable=grouping_variable)
        assert isinstance(ice_plot_result, list)
        assert isinstance(ice_plot_result[0].figure(), matplotlib.pyplot.Figure)

    matplotlib.pyplot.close("all")

def test_binary_response_scale():
    train = h2o.upload_file(pyunit_utils.locate("smalldata/titanic/titanic_expanded.csv"))
    y = "survived"

    # get at most one column from each type
    cols_to_test = []
    for col, typ in train.types.items():
        for ctt in cols_to_test:
            if typ == train.types[ctt] or col == y:
                break
        else:
            cols_to_test.append(col)

    gbm = H2OGradientBoostingEstimator(seed=1234, model_id="my_awesome_model")
    gbm.train(y=y, training_frame=train)

    assert isinstance(gbm.ice_plot(train, 'title', binary_response_scale="logodds").figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'age').figure(), matplotlib.pyplot.Figure)
    matplotlib.pyplot.close("all")

    try:
        gbm.ice_plot(train, 'title', binary_response_scale="invalid_value")
    except ValueError as e:
        assert str(e) == "Unsupported value for binary_response_scale!"

    y = "fare"
    gbm = H2OGradientBoostingEstimator(seed=1234, model_id="my_awesome_model")
    gbm.train(y=y, training_frame=train)

    try:
        gbm.ice_plot(train, 'title', binary_response_scale="logodds")
    except ValueError as e:
        assert str(e) == "binary_response_scale cannot be set to 'logodds' value for non-binomial models!"


def test_show_pdd():
    train = h2o.upload_file(pyunit_utils.locate("smalldata/titanic/titanic_expanded.csv"))
    y = "fare"

    # get at most one column from each type
    cols_to_test = []
    for col, typ in train.types.items():
        for ctt in cols_to_test:
            if typ == train.types[ctt] or col == y:
                break
        else:
            cols_to_test.append(col)

    gbm = H2OGradientBoostingEstimator(seed=1234, model_id="my_awesome_model")
    gbm.train(y=y, training_frame=train)

    assert isinstance(gbm.ice_plot(train, 'title').figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'title', show_pdp=True).figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'title', show_pdp=False).figure(), matplotlib.pyplot.Figure)

    assert isinstance(gbm.ice_plot(train, 'age').figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'age', show_pdp=True).figure(), matplotlib.pyplot.Figure)
    assert isinstance(gbm.ice_plot(train, 'age', show_pdp=False).figure(), matplotlib.pyplot.Figure)
    matplotlib.pyplot.close("all")




pyunit_utils.run_tests([
    test_binary_response_scale,
    test_show_pdd,
    test_display_mode
])