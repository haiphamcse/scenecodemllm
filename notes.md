
idea_3i: original Perceiver Encoder + Qwen

idea_3i_scene_graph: i think it's adding small LM decoder head to regress scene graph

idea_3i_scene_graph_v3toon_latentonly: Perceiver Encoder + Qwen but use for scene graph generation. Using V3-Toon format scene graph
idea

idea_3i_scene_graph_v3toon_video: same as above but also adding video as input

ca1m_metric_toon_latentonly: Perceiver + Qwen but using CA1M dataset with Toon format

idea_3i_toon_ca1m_metric_vg: VG-LLM Encoder + Qwen but using CA1M dataset with Toon format

idea_3i_vg_ca1m_metric_perceiver: Perceiver + Qwen but using CA1M dataset with VG-LLM Json format

How to rename? -> Brand it to idea_4X

idea_4a: Perceiver + Qwen for generating scene graph using Toon format

idea_4b: Perceiver + Qwen for generating scene graph using VG-LLM JSON format

idea_4c: VG-LLM adapter + Qwen for generating scene graph using Toon format

