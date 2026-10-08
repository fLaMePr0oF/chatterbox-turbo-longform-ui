# Contributing

Contributions are welcome.

## Suggested workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Keep Turbo-specific and V3-specific model behaviour separate where their APIs differ.
4. Preserve the shared long-form editor behaviour unless a change is intentionally made to both apps.
5. Syntax-check Python changes before opening a pull request.
6. Describe the change and how it was tested in the pull request.

## Design principle

The two front ends should remain visually and functionally aligned wherever practical, while preserving controls and behaviours that are specific to each Chatterbox model.
