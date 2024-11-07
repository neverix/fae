import jax
from pathlib import Path
from .quant import QuantMatrix
import jax.numpy as jnp
import qax
import orbax.checkpoint as ocp
import shutil


def save_quantized(a):
    vals, treedef = jax.tree.flatten(
        a, is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue)
    )
    treedef = jax.tree.unflatten(treedef, [jnp.zeros((1,)) for _ in vals])
    new_vals = []
    for val in vals:
        if isinstance(val, QuantMatrix):
            new_vals.append(("qm", val.quants, val.scales, val.use_approx, str(val.orig_dtype), val.use_kernel, val.mesh_and_axis))
        else:
            new_vals.append(val)
    vals = new_vals
    return treedef, vals

def load_quantized(restored_treedef, restored_vals):
    _, restored_treedef = jax.tree.flatten(
        restored_treedef, is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue)
    )
    new_vals = []
    for val in restored_vals:
        if isinstance(val, list) and val[0] == "qm":
            quants, scales, use_approx, orig_dtype, use_kernel, mesh_and_axis = val[1:]
            new_vals.append(QuantMatrix(
                quants=quants, scales=scales,
                use_approx=use_approx, orig_dtype=orig_dtype,
                use_kernel=use_kernel, mesh_and_axis=mesh_and_axis))
        else:
            new_vals.append(val)
    restored_a = jax.tree.unflatten(restored_treedef, new_vals)
    return restored_a


def restore_array(path):
    with jax.default_device(jax.devices("cpu")[0]):
        checkpointer = ocp.PyTreeCheckpointer()
        vals_meta = checkpointer.metadata(path)
        sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        restore_args = ocp.checkpoint_utils.construct_restore_args(
            vals_meta,
            sharding_tree=jax.tree.map(lambda _: sharding, vals_meta)
        )
        restored_vals = checkpointer.restore(path, restore_args=restore_args)
    return restored_vals


def save_thing(value, og_path):
    og_path = Path(og_path).resolve()
    treedef, vals = save_quantized(value)
    path = ocp.test_utils.erase_and_create_empty(og_path)
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        checkpointer.save(path / "treedef", treedef)
        checkpointer.save(path / "vals", vals)
    except ValueError:
        shutil.rmtree(og_path)
        raise


def load_thing(path):
    path = Path(path).resolve()
    treedef = restore_array(path / "treedef")
    vals = restore_array(path / "vals")
    return load_quantized(treedef, vals)