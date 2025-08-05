def evaluate_on_test_env(model, test_env):
    """
    在测试环境中评估模型，并返回测试结果
    """
    obs = test_env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, infos = test_env.step(action)

    portfolio_value = infos[0]['portfolio_value']
    mean_rank_ic = infos[0]['mean_rank_ic']
    transaction_cost_result = infos[0]['transaction_cost_result']

    print(
        f"test_total_asset: {portfolio_value} mean_rank_ic: {mean_rank_ic} transaction_cost_result: {transaction_cost_result}")

    return portfolio_value

def run_test_evaluation(model, env_test):
    obs = env_test.reset()
    done = False

    max_len=0

    shift_rewards = []
    dates = []
    portfolio_values = []
    returns = []
    assets_scores = []
    while not done:
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, done, infos = env_test.step(action)

        shift_rewards.append(infos[0]['portfolio_shift_reward'])
        dates.append(infos[0]['date'])
        portfolio_values.append(infos[0]['portfolio_value'])
        returns.append(infos[0]['returns'])
        assets_scores.append(infos[0]['scores'])
        max_len+=1
    
    return dates, portfolio_values, shift_rewards, infos
