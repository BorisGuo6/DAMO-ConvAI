import os

import torch
from torch import nn
from transformers.modeling_utils import (
    ModuleUtilsMixin, PushToHubMixin,
    unwrap_model, get_parameter_dtype,
)
from transformers.utils import logging, WEIGHTS_NAME
from typing import Union, Optional, Callable

try:
    from transformers.utils import is_offline_mode, is_remote_url
except ImportError:
    def is_offline_mode():
        return False
    def is_remote_url(url):
        return url.startswith("http://") or url.startswith("https://")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

logger = logging.get_logger(__name__)

# Constants for model weights
TF_WEIGHTS_NAME = "model.ckpt"
TF2_WEIGHTS_NAME = "tf_model.h5"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"


class PushToHubFriendlyModel(nn.Module, ModuleUtilsMixin, PushToHubMixin):
    def __init__(self):
        super().__init__()

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            save_config: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        dtype = get_parameter_dtype(model_to_save)
        self.pretrain_model.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        self.pretrain_model.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            self.pretrain_model.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    def load(self, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Adopted and simplified from transformers.modeling_utils from_pretrained,
        but more similiar to load_state_dict(load the weight from anywhere into a create model).
        """
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        force_download = kwargs.pop("force_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        from_pt = not (from_tf | from_flax)

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no file named {[WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + '.index', FLAX_WEIGHTS_NAME]} found in "
                        f"directory {pretrained_model_name_or_path} or `from_tf` and `from_flax` set to False."
                    )
                resolved_archive_file = archive_file
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                resolved_archive_file = archive_file
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = pretrained_model_name_or_path + ".index"
                resolved_archive_file = archive_file
            else:
                # Download from Hugging Face Hub
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                else:
                    filename = WEIGHTS_NAME

                if hf_hub_download is not None:
                    try:
                        resolved_archive_file = hf_hub_download(
                            repo_id=pretrained_model_name_or_path,
                            filename=filename,
                            revision=revision,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            local_files_only=local_files_only,
                        )
                    except Exception as err:
                        logger.error(err)
                        raise EnvironmentError(
                            f"Can't load weights for '{pretrained_model_name_or_path}'. "
                            f"Make sure it's a correct model identifier on Hugging Face Hub."
                        )
                else:
                    raise EnvironmentError(
                        f"Can't download weights. Please install huggingface_hub: pip install huggingface_hub"
                    )

            logger.info(f"loading weights file {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # load pt weights
        if from_pt:
            if state_dict is None:
                try:
                    state_dict = torch.load(resolved_archive_file, map_location="cpu")
                except Exception:
                    raise OSError(
                        f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                        f"at '{resolved_archive_file}'"
                        "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                    )
        self.load_state_dict(state_dict, strict=True)
