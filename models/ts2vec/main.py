import time
import datetime
from .ts2vec import TS2Vec


def train_TS2Vec(args, train_loader, valid_loader):
    """ Train TS2Vec and return best model. 
    
    :param args: arguemtns
    :type args: dictionary
    :param train_loader: train loader
    :type train_loader: iterable-style datasets
    :param valid_loader: valid loader
    :type valid_loader: iterable-style datasets
    :return: best trained model
    :rtype: model
    
    example
        >>> best_model = train_TS2Vec(args, train_loader, valid_loader)
    """
    
    # set configuration for training model
    config = dict(
        batch_size=args['batch_size'],
        lr=args['lr'],
        output_dims=args['repr_dims']
    )

    # build & fit TS2Vec model
    t = time.time()
    
    model = TS2Vec(
        input_dims=args['input_dims'],
        device=args['device'],
        **config
    )
    
    print("Start Training...")
    loss_log, best_model = model.fit(
        train_loader,
        valid_loader,
        n_epochs=args['epochs'],
        verbose=True
    )
    
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print("Finished.")

    return best_model

def encode_TS2Vec(args, test_loader, model):
    """ Encode representations from trained model. 
    
    :param args: arguemtns
    :type args: dictionary
    :param test_loader: test loader
    :type test_loader: iterable-style datasets
    :return: representation vector
    :rtype: numpy array
    
    example
        >>> result_repr = train_TS2Vec(args, test_loader, model)
    """
    # build model
    config = dict(
        batch_size=args['batch_size'],
        lr=args['lr'],
        output_dims=args['repr_dims']
    )
    
    base_model = TS2Vec(
        input_dims=args['input_dims'],
        device=args['device'],
        **config
    )
    
    # load best model
    base_model.net.load_state_dict(model.state_dict())
    
    # get representation
    result_repr = base_model.encode(test_loader)
    
    return result_repr
