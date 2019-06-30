def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    # DQN
    parser.add_argument('--dueling_dqn', action='store_true', help='whether to use Dueling DQN')
    parser.add_argument('--prioritized_dqn', action='store_true', help='whether to use Prioritized Replay buffer')
    
    return parser
