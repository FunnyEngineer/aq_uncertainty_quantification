import jax.numpy as jnp
import jax
import pdb

# deterministic version
@jax.vmap
def mse_loss(preds, targets):
  return jnp.mean((preds.flatten() - targets)**2)

@jax.jit
def apply_model(state, x, y):
    def loss_fn(params):
        y_hat = state.apply_fn({'params': params}, x)         # make forward pass
        loss = mse_loss(y_hat, y).mean()
        return loss, y_hat
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, y_hat), grads = grad_fn(state.params)
    return state, loss, grads

def train_one_epoch(state, training_generator):
    train_loss = []
    for i, (x, y) in enumerate(training_generator):
        state, loss, grads = apply_model(state, x, y)

        train_loss.append(loss.item())

        state = state.apply_gradients(grads=grads)
    train_loss = jnp.mean(jnp.array(train_loss))
    return state, train_loss

# probabilistic version

@jax.jit
def update_model(state, x, y):
    def loss_fn(params, x, y):
        # Apply the neural network model, and obtain a prediction
        y_hat = state.apply_fn({'params': params}, x) 

        # Compute the loss function for this batch of data
        # In this instance, a simple l2 loss, averaged over the batch
        total = y_hat.log_prob(y)
        import pdb; pdb.set_trace()
        return -jnp.mean(total)
    # Computes the gradients of the model
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)

    return state, loss, grads

def train_one_epoch_prob(state, training_generator):
    train_loss = []
    for i, (x, y) in enumerate(training_generator):
        state, loss, grads = update_model(state, x, y)
        train_loss.append(loss.item())

        state = state.apply_gradients(grads=grads)
    train_loss = jnp.mean(jnp.array(train_loss))
    return state, train_loss
