"""
Evaluation functions
"""


def dummy_evaluation_func(state):
    return 0.0


def distance_evaluation_func(state):
    player = state.get_current_player()
    info = state.get_info()
    score = 0.0 
    for p, info_p in info.items():
        if p == player:
            score -= info_p["max_distance"]
        else:
            score += info_p["max_distance"]
    return score


def detailed_evaluation_func(state):
    # TODO
    '''
    活四、冲四、活三、冲三、活二的数目
    最远棋子距离棋盘中心的相对距离（最小为0，最大为1）
    '''
    player = state.get_current_player()
    info = state.get_info()
    score = 0.0
    for p, info_p in info.items():
        distance = info_p["max_distance"]
        live_four = info_p["live_four"]
        four = info_p["four"]
        live_three = info_p["live_three"]
        three = info_p["three"]
        live_two = info_p["live_two"]
        if p == player:
            score += (40000*live_four + 5000*four + 1000*live_three + 100 * three + 50 *live_two - 500 * distance)/50000
        else:
            score -= (40000*live_four + 100*four + 1000*live_three + 100 * three + 50 *live_two - 500 * distance)/50000
    return score


def get_evaluation_func(func_name):
    if func_name == "dummy_evaluation_func":
        return dummy_evaluation_func
    elif func_name == "distance_evaluation_func":
        return distance_evaluation_func
    elif func_name == "detailed_evaluation_func":
        return detailed_evaluation_func
    else:
        raise KeyError(func_name)
