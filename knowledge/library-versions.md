# Library Versions Guideline

## Purpose
This document serves as the authoritative reference for the latest stable versions of essential libraries. When generating code, LLMs must always use the versions specified here for the listed libraries to ensure compatibility, security, and adherence to best practices. For libraries not mentioned in this guideline, version selection is left to the LLM's discretion.

## Essential Libraries & Their Latest Versions

- **py-trees**: `2.3.0`
  - [PyPI Page](https://pypi.org/project/py-trees/)
  - [Documentation](https://py-trees.readthedocs.io/en/devel/)

- **openai**: `1.65.1`
  - [GitHub Repository](https://github.com/openai/openai-python)
  - [PyPI Page](https://pypi.org/project/openai/)

- **groq**: `0.18.0`
  - [GitHub Repository](https://github.com/groq/groq-python)
  - [PyPI Page](https://pypi.org/project/groq/)

## Guidelines for Version Selection

- **Mandatory Version Usage**:  
  - Always reference and use the versions specified for the essential libraries above in generated code.
  - Do not override these versions unless explicitly required by a project's context.

- **Non-Mentioned Libraries**:  
  - For libraries not included in this document, LLMs may choose the most appropriate version based on context and current stable releases.

- **Conflict Resolution**:  
  - If a project-specific dependency conflicts with the versions specified here, carefully document the decision to deviate and ensure that the rationale is well justified.

## Conclusion
This guideline acts as a version map for critical libraries, ensuring that generated code is both up-to-date and secure. Always refer to the official resources provided above to verify version information and update this document as new stable releases become available.
