from constants import EXPERIMENTS_ALL, EXPERIMENTS_UNET_ALL, UNET_CHECKPOINT
from RaysAttackExperiment import RaysAttackExperiment
from ADBAAttackExperiment import ADBAAttackExperiment
from SquareAttackLinfExperiment import SquareAttackLinfExperiment
from SurfreeAttackExperiment import SurfreeAttackExperiment, DEFAULT_SURFREE_CONFIG


def main():
    epsilon = [255/255, 64/255, 32/255, 16/255, 8/255, 4/255]

    # Run all Rays attacks
    for eps in epsilon:
        RaysAttackExperiment(experiments_config=EXPERIMENTS_ALL, epsilon_max=eps).run_all()

    # Run all ADBA attacks
    for eps in epsilon:
        ADBAAttackExperiment(experiments_config=EXPERIMENTS_ALL, epsilon_max=eps).run_all()

    # Run all Square attacks
    for eps in epsilon:
        SquareAttackLinfExperiment(experiments_config=EXPERIMENTS_ALL, epsilon_max=eps).run_all()

    # Run all Surfree attacks
    for eps in epsilon:
        SurfreeAttackExperiment(experiments_config=EXPERIMENTS_ALL, surfree_config=DEFAULT_SURFREE_CONFIG).run_all()


if __name__ == "__main__":
    main()