def eval_nb_samples(f, rounds, **args):
    cnt = 0
    for _ in range(rounds):
        cnt += f(**args)['nb_samples']
    return cnt / rounds

if __name__ == '__main__':
    def f(x):
        return {'nb_samples': x}
    cnt = eval_nb_samples(f, rounds=10, x=3)
    print(cnt)