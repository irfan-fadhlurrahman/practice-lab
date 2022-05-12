from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor

def regressors(seed=0):
    regs = {}

    regs['RidgeCV'] = RidgeCV()
    regs['LassoCV'] = LassoCV()
    regs['ElasticNetCV'] = ElasticNetCV()
    regs['DecisionTreeRegressor'] = DecisionTreeRegressor(
        random_state=seed, max_depth=3
    )
    regs['RandomForestRegressor'] = RandomForestRegressor(
        random_state=seed, max_depth=3
    )
    regs['HistGradientBoostingRegressor'] = HistGradientBoostingRegressor(
        random_state=seed, max_depth=3
    )
    regs['XGBRegressor'] = XGBRegressor(
        random_state=seed, max_depth=3
    )
    regs['VotingRegressor'] = VotingRegressor(
        estimators=[
           ('dt', regs['RandomForestRegressor']),
           ('rf', regs['HistGradientBoostingRegressor']),
           ('gb', regs['XGBRegressor'])
        ], voting='hard',
           weights=[1, 2, 3]
    )
    return regs
