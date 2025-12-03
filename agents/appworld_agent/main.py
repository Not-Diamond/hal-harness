import os

from appworld import cli
from appworld.common.path_store import path_store
from appworld.common.utils import read_file, write_file
from appworld.evaluator import evaluate_dataset


current_directory = os.path.dirname(os.path.abspath(__file__))
path_store.update_root(current_directory)


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # assert required keys
    required_keys = ["model_name", "method_name", "provider"]
    for required_key in required_keys:
        assert required_key in kwargs, f"{required_key} is required"

    # Validate method_name
    assert isinstance(kwargs['method_name'], str), f"method_name must be a string, got {type(kwargs['method_name'])}"

    # Get valid agent types from appworld configs
    valid_agent_types = [
        d for d in os.listdir(path_store.experiment_configs)
        if os.path.isdir(os.path.join(path_store.experiment_configs, d))
        and "legacy" not in d
        and "agent" in d
    ]
    assert kwargs['method_name'] in valid_agent_types, (
        f"method_name '{kwargs['method_name']}' is not a valid agent type. "
        f"Valid options: {', '.join(valid_agent_types)}"
    )

    # create a dataset file with only passed task
    task_ids = list(input.keys())
    dataset_name = "sample"
    dataset_file_path = os.path.join(current_directory, "data", "datasets", f"{dataset_name}.txt")
    write_file("\n".join(task_ids), dataset_file_path)
    # create a sample config file for the experiment
    reference_experiment_name = f"{kwargs['method_name']}/{kwargs["provider"]}/{kwargs['model_name']}/test_challenge"
    reference_experiment_config_file_path = os.path.join(
        path_store.experiment_configs, f"{reference_experiment_name}.jsonnet"
    )
    reference_experiment_config = read_file(reference_experiment_config_file_path)
    actual_experiment_config = reference_experiment_config.replace("test_challenge", dataset_name)
    actual_experiment_name = "output"
    actual_experiment_config_file_path = os.path.join(
        path_store.experiment_configs, f"{actual_experiment_name}.jsonnet"
    )
    write_file(actual_experiment_config, actual_experiment_config_file_path)
    # run the experiment
    cli.run(
        experiment_name=actual_experiment_name,
        model_name=None,
        agent_name=None,
        dataset_name=None,
        dry_run=False,
        task_id=None,
        with_evaluation=False,
        override=None,
        clear_first=False,
        num_processes=1,
        process_index=None,
        with_setup=False,
        root=current_directory,
    )
    # evaluate the results
    evaluation = evaluate_dataset(
        experiment_name=actual_experiment_name,
        dataset_name=dataset_name,
        suppress_errors=True,
        include_details=True,
    )
    return {task_ids[0]: "Completed", "evaluation": evaluation}
