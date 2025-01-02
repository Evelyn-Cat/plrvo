## for all:

we shall run scripts in run scripts/todo.*.sh.

for plorv noise:
    [scripts/todo.run_plrvo.sh] we use plrvo_transoformers to load plrvo/{configs_idx}.json and continue.
for gaussian noise:
    1 [scripts/todo.run_gaussian.sh] we can plrvo_transoformers to load plrvo/{configs_idx}.json and continue.
    2 [scripts/todo.run_gaussian_manually.sh]we can private_transformers as long as we set (1) noise_mutiplier and (2) clipping value.


