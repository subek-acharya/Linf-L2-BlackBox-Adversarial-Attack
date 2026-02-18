from constants import EXPERIMENTS
from RaysAttackExperiment import RaysAttackExperiment
from ADBAAttackExperiment import ADBAAttackExperiment
from SquareAttackLinfExperiment import SquareAttackLinfExperiment


def main():
    epsilon = [255/255, 64/255, 32/255, 16/255, 8/255, 4/255]

    # Run all Rays attacks first
    for eps in epsilon:
        RaysAttackExperiment(experiments_config=EXPERIMENTS, epsilon_max=eps).run_all()

    # Run all ADBA attacks second
    for eps in epsilon:
        ADBAAttackExperiment(experiments_config=EXPERIMENTS, epsilon_max=eps).run_all()

    # Run all Square attacks last
    for eps in epsilon:
        SquareAttackLinfExperiment(experiments_config=EXPERIMENTS, epsilon_max=eps).run_all()


if __name__ == "__main__":
    main()