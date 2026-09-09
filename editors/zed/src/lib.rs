
struct Resin;

impl zed_extension_api::Extension for Resin {
    fn new() -> Self {
        Self
    }

    fn language_server_command(
        &mut self,
        id: &zed_extension_api::LanguageServerId,
        worktree: &zed_extension_api::Worktree,
    ) -> zed_extension_api::Result<zed_extension_api::Command> {
        let settings = zed_extension_api::settings::LspSettings::for_worktree(id.as_ref(), worktree)?;
        let mut args = vec!["--lsp".into(), worktree.root_path()];
        let mut env = worktree
            .shell_env()
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        let configured = settings.binary.and_then(|binary| {
            if let Some(arguments) = binary.arguments {
                args = arguments;
            }
            if let Some(overrides) = binary.env {
                env.extend(overrides);
            }
            binary.path
        });
        let command = configured.or_else(|| worktree.which("resin")).ok_or("Install resin with nix-shell --run 'cargo install --path . --locked', or configure lsp.resin-lsp.binary.path in Zed.")?;
        Ok(zed_extension_api::Command {
            command,
            args,
            env: env.into_iter().collect(),
        })
    }

    fn language_server_initialization_options(
        &mut self,
        id: &zed_extension_api::LanguageServerId,
        worktree: &zed_extension_api::Worktree,
    ) -> zed_extension_api::Result<Option<zed_extension_api::serde_json::Value>> {
        Ok(zed_extension_api::settings::LspSettings::for_worktree(id.as_ref(), worktree)?.initialization_options)
    }
}

zed_extension_api::register_extension!(Resin);
