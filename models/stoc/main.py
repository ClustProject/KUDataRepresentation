import time
import datetime
from .modules.trainer import Trainer


def train_STOC(args, train_loader, valid_loader):
    """ Train STOC and return best model. 
    
    :param args: arguemtns
    :type args: dictionary
    :param train_loader: train loader
    :type train_loader: iterable-style datasets
    :param valid_loader: valid loader
    :type valid_loader: iterable-style datasets
    :return: best trained model
    :rtype: model
    
    example
        >>> best_model = train_STOC(args, train_loader, valid_loader)

    """
    # set configuration for training model

    config = dict(
        batch_size=args['batch_size'],
        lr=args['lr'],
        output_dim=args['repr_dim'],
        input_dim=args['input_dim'],
        device=args['device'], 
        patience=args['patience'], 
    )

    # build & fit STOC model
    t = time.time()
    
    trainer = Trainer(
        **config
    )
    
    print("Start Training...")
    best_model = trainer.fit(
        train_loader,
        valid_loader,
        num_epochs=args['num_epochs'],
        verbose=True
    )
    
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print("Finished.")

    return best_model


def encode_STOC(args, test_loader, best_model):
    """ Encode representations from trained model. 
    
    :param args: arguemtns
    :type args: dictionary
    :param test_loader: test loader
    :type test_loader: iterable-style datasets
    :return: representation vector
    :rtype: numpy array
    
    example
        >>> result_repr = encode_STOC(args, test_loader, model)
    """
    # build model
    config = dict(
        batch_size=args['batch_size'],
        lr=args['lr'],
        output_dim=args['repr_dim'],
        input_dim=args['input_dim'],
        device=args['device'], 
        patience=args['patience']
    )
    
    trainer = Trainer(
        **config
    )
    
    # load best model
    trainer.model.load_state_dict(best_model.state_dict())
    
    # get representation
    result_repr = trainer.encode(test_loader)

    return result_repr

