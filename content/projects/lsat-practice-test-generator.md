---
title: "LSAT Practice Test Generator - Open Source"
date: 2024-09-01
draft: false
weight: 20
---
### Objective
To generate practice tests with real LSAT questions from old tests using natural language prompts.

### Approach
Developed a novel algorithm to create hyperdimensional contextual encodings of LSAT questions. This allows for sophisticated understanding and generation of relevant test content.

<img src="/images/lsat-screenshot.png" alt="Generated LSAT Practice Test" class="w-full h-auto rounded-lg shadow-lg my-8">

```python
# Python snippet demonstrating the core of the core of the encoder
# This would typically be part of a larger codebase, but is included here
# to showcase the technical approach.
class Encoder:
    ### --^^-- initialization --^^-- ###
    def generate_orthogonal_roles(self, num_roles=3):
        random_matrix = np.random.randn(self.output_dim, num_roles)
        ortho_matrix = orth(random_matrix)
        ortho_matrix = ortho_matrix[:, :num_roles]
        roles = {
            "stimulus": ortho_matrix[:, 0],
            "prompt": ortho_matrix[:, 1],
        }
        if num_roles > 2:
            roles["explanation"] = ortho_matrix[:, 2]
        return roles

    def simhash_projection(self, embedding: np.ndarray) -> np.ndarray:
        real_hdv = np.zeros(self.output_dim)
        for i, value in enumerate(embedding):
            hash_index = murmurhash3_32(i, positive=True) % self.output_dim
            real_hdv[hash_index] += value
        return real_hdv

    def project_to_hdvs(self, embeddings: np.ndarray) -> np.ndarray:
        hdvs = np.array([self.simhash_projection(emb) for emb in embeddings])
        return hdvs

    def bind(self, role_hdv: np.ndarray, value_hdv: np.ndarray) -> np.ndarray:
        return self.circular_convolve(role_hdv, value_hdv)

    def bundle(self, bound_hdvs: list) -> np.ndarray:
        bundled_hdv = np.sum(bound_hdvs, axis=0)
        return bundled_hdv

    def generate_question_hdv_from_json(
            self,
            json_obj: dict,
            roles: dict,
            weights: dict[str, float] = None
    ) -> np.ndarray:
        field_values = {
            "stimulus": json_obj.get("stimulus", ""),
            "prompt": json_obj.get("prompt", ""),
            "explanation": json_obj.get("explanation", ""),
        }
        sentences = list(field_values.values())
        value_embeddings = self.get_sentence_embeddings(sentences)
        value_hdvs = self.project_to_hdvs(value_embeddings)
        bound_hdvs = [self.bind(roles[key], value_hdvs[i]) for i, key in enumerate(field_values.keys())]
        question_hdv = self.bundle(bound_hdvs)
        return question_hdv
```

### Results & Impact
* Allows for use of natural language prompts to generate practice tests with real LSAT questions from old tests.
* Open Source project, contributing to the broader ML community.
