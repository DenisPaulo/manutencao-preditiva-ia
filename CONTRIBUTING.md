# Contribuindo

Obrigado pelo interesse em **manutencao-preditiva-ia**! Este é um projeto educacional/de portfólio (modelo de manutenção preditiva com XGBoost e SHAP, e painel em Streamlit).

## Como sugerir melhorias

- **Ideias e problemas:** abra uma [issue](https://github.com/DenisPaulo/manutencao-preditiva-ia/issues) explicando o que você observou ou propõe.
- **Código ou documentação:** faça um fork, crie uma branch, e abra um Pull Request descrevendo a mudança. Prefira PRs pequenos e focados.

## Como rodar localmente

Para rodar localmente, siga as instruções do [README](README.md) (instalação com `requirements.txt`; use `requirements-dev.txt` para notebooks).

## Antes de abrir o PR

Antes de abrir o PR, rode os testes e o lint:

```bash
pip install pytest ruff
ruff check .
pytest -q
```

Não inclua segredos, credenciais ou dados pessoais no código, nos commits ou nos exemplos (veja a [Política de Segurança](SECURITY.md)).

## Padrão de commits

Mensagens curtas e no imperativo, em português, com um prefixo opcional:

- `feat:` nova funcionalidade
- `fix:` correção de bug
- `docs:` documentação
- `chore:` manutenção
