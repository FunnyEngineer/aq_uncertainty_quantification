import jax
import jax.numpy as jnp

# deterministic version

@jax.vmap
def mse_loss(preds, targets):
  return jnp.mean((preds.flatten() - targets)**2)

def eval(state, testing_generator):
    eval_loss = []
    for i, (x,y) in enumerate(testing_generator):
        # variables = model.init(rng, x)
        def loss_fn(params):
            # make forward pass
            y_hat = state.apply_fn({'params': params}, x)

            loss = mse_loss(y_hat, y).mean()
            eval_loss.append(loss.primal.item())
            return loss
        grads = jax.grad(loss_fn)(state.params)
        # eval_loss.append(grads)
    eval_loss = jnp.mean(jnp.array(eval_loss))
    return eval_loss


# probabilistic version

def eval_prob(state, testing_generator):
    eval_loss = []
    for i, (x, y) in enumerate(testing_generator):
        # variables = model.init(rng, x)
        def loss_fn(params):
            # make forward pass
            q = state.apply_fn({'params': params}, x)

            loss = -jnp.mean(q.log_prob(y[:]))
            eval_loss.append(loss.primal.item())
            return loss
        grads = jax.grad(loss_fn)(state.params)
        # eval_loss.append(grads)
    eval_loss = jnp.mean(jnp.array(eval_loss))
    return eval_loss