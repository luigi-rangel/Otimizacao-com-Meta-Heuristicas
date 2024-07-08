from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from lib.utils.experiment import Experiment

class Test:
    def __init__(self, n, experiments, n_samples, n_gens, title,
                 keep_x=False, keep_best=True):
        self.experiments = [
            Experiment(experiments[i].tag,
                       experiments[i].algorithm,
                       experiments[i].params) for i in range(len(experiments))
        ]
        self.n = n
        self.n_samples = n_samples
        self.n_gens = n_gens
        self.title = title
        self.keep_x = keep_x
        self.keep_best = keep_best

    def run(self):
        for i in range(len(self.experiments)):
            config = self.experiments[i]
            for _ in range(self.n_samples):
                config.run(self.n_gens, self.keep_x, self.keep_best)
            config.x_histories = np.array(config.x_histories)
            config.apt_x_histories = np.array(config.apt_x_histories)
            config.best_histories = np.array(config.best_histories)
            config.apt_best_histories = np.array(config.apt_best_histories)
            config.fin_res = config.apt_best_histories[:, -1]

        for i in range(len(self.experiments)):
            config = self.experiments[i]
            res, pvalue = self.run_ttest()
            print(f'{self.n:5}', end=' | ')
            print(f'{self.title}', end=' | ')
            print(f'{config.tag:>9}', end=' | ')
            print(f'{config.fin_res.mean():.2e}', end=' | ')
            print(f'{config.fin_res.std():.2e}', end=' | ')
            print(f'{config.fin_res.min():.2e}', end=' | ')
            print(f'{config.fin_res.max():.2e}', end=' | ')
            if i == 0:
                print(f'p={pvalue:5.3e}')
            else: print(f' {res:>10}')
        
        return self

    def plot_mean_convergence(self, ax, fmts=['-']):
        its = np.arange(1, self.n_gens + 1)
        for i in range(len(self.experiments)):
            ax.plot(its, np.mean(self.experiments[i].apt_best_histories, 0),
                    fmts[i % len(fmts)], label=self.experiments[i].tag)
        ax.legend()
        ax.set_ylabel('Aptidão média')
        ax.set_title('Convergência')

    def plot_median_convergence(self, ax):
        its = np.arange(1, self.n_gens + 1)
        for i in range(len(self.experiments)):
            ax.plot(its, np.median(self.experiments[i].apt_best_histories, 0),
                    label=self.experiments[i].tag)
        ax.legend()
        ax.set_ylabel('Aptidão mediana')
        ax.set_title('Convergência')

    def boxplot_results(self, ax, gen, colors = ['blue', 'red']):
        results = np.array(
            list(map(lambda x: x.apt_best_histories[:, gen], self.experiments))).transpose()
        df = pd.DataFrame(results, columns=self.get_columns())
        bplot = df.boxplot(column=self.get_columns(), ax=ax,
                        notch=True, grid=False, fontsize=10,
                        return_type='dict', patch_artist=True,
                        showmeans=True, meanprops={'marker': 'D',
                                                   'markerfacecolor': 'black',
                                                   'markeredgecolor': 'black',
                                                   'markersize': 4
                                                   },
                        boxprops={'color': 'black'},
                        whiskerprops={'color': 'black'},
                        medianprops={'color': 'green'},
                        capprops={'color': 'black'})
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        ax.legend(handles=[Line2D([0], [0], marker='D',
                                  color='black', label='Fitness Médio',
                                  markerfacecolor='black', linestyle='None')])
        ax.set_ylabel('Aptidão')
        ax.set_title(f'{gen + 1}ª geração')

    def get_columns(self):
        return list(map(lambda x: x.tag, self.experiments))

    def run_ttest(self):
        null_hyp = self.experiments[0].fin_res
        alt_hyp = self.experiments[1].fin_res
        test_eq_avg = stats.ttest_ind(null_hyp, alt_hyp)
        pvalue = test_eq_avg.pvalue
        res = 'H0 reject'if pvalue < 0.05 else 'H0 accept'
        return res, pvalue