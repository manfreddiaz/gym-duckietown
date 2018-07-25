# Record and Replay API

Each episode contains a list of dictionaries with agent, environment and internal structure of the environment.
```python
self._episodes_current.append({
    'agent': np.array([
        observation,
        action
    ]),
    'env': np.array([
        next_observation,
        reward,
        done,
        info
    ]),
    'hidden': np.array([
        unwrapped_env.cur_pos,
        unwrapped_env.cur_angle,
        unwrapped_env.step_count
    ])
})
```

All episodes are sequentially stored in a pickle file and can simply be read with the following code snippet:

```python
    try:
        current_episode = pickle.load(self.recording_file)
    except EOFError:
        print('no more episodes')
```

Check the algorithms folder for sample of usage.