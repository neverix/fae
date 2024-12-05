import jax
from pathlib import Path
import orbax.checkpoint as ocp
import shutil
from loguru import logger


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
    logger.info("Creating values for saving")
    with jax.default_device(jax.devices("cpu")[0]):
        # treedef, vals = save_quantized(value)
        logger.info("Saving values")
        og_path = Path(og_path).resolve()
        path = ocp.test_utils.erase_and_create_empty(og_path)
        checkpointer = ocp.PyTreeCheckpointer()
        try:
            checkpointer.save(path / "everything", value)
        except ValueError:
            shutil.rmtree(og_path)
            raise


def load_thing(path):
    path = Path(path).resolve()
    with jax.default_device(jax.devices("cpu")[0]):
        return restore_array(path / "everything")