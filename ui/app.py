"""Development console.

Run it:

    streamlit run ui/app.py

Draws screens and nothing else. Every call to the system goes through
ui/backend.py, so this file never imports from domain/ or adapters/ - which
is what keeps "wire a screen to real code" a change in one place.

Sections marked STUB do not run anything. That labelling is the point: a
console that quietly fabricates an answer teaches you the system works when
it does not.

There is no login. It was left out deliberately rather than faked - real
authentication is users, hashed passwords and sessions, and a name field
pretending to be a login is the kind of thing that gets mistaken for
security later.
"""

from __future__ import annotations

import streamlit as st

from backend import (
    AgentDraft,
    Connection,
    ask,
    classify_preview,
    distinct_count,
    introspect,
    save_agent,
    suggested_role_name,
)

st.set_page_config(page_title="Agent console", page_icon="🔐", layout="wide")

# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

st.session_state.setdefault("schema", None)      # dict[str, TableSchema] | None
st.session_state.setdefault("agents", [])        # list[AgentDraft]
st.session_state.setdefault("chat", [])          # list[dict]
st.session_state.setdefault("error", None)


def stub_note(text: str) -> None:
    """Mark a section that does not run anything yet."""
    st.caption(f"⚠️ **Stub** — {text}")


# --------------------------------------------------------------------------
# sidebar: the database connection
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Database")

    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", value=55432, step=1, format="%d")
    database = st.text_input("Database", value="library_dev")

    # The role matters more than it looks. information_schema reports only
    # what the connected role may read, so switching between app_catalog and
    # app_circulation changes the table list below - which is the security
    # design, visible.
    role = st.selectbox(
        "Connect as",
        ["dev / dev", "app_catalog / dev_catalog", "app_circulation / dev_circulation", "custom"],
        help="Each agent role sees only its own tables. Switch and reconnect to watch it.",
    )
    if role == "custom":
        user = st.text_input("User", value="dev")
        password = st.text_input("Password", value="dev", type="password")
    else:
        user, password = [part.strip() for part in role.split("/")]

    connection = Connection(
        host=host, port=int(port), database=database, user=user, password=password
    )

    if st.button("Connect", type="primary", use_container_width=True):
        try:
            st.session_state.schema = introspect(connection)
            st.session_state.error = None
        except Exception as e:
            st.session_state.schema = None
            st.session_state.error = str(e)

    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.schema is not None:
        st.success(f"{user} — {len(st.session_state.schema)} readable tables")
    else:
        st.info("Not connected.")

    st.divider()
    st.caption(
        "Start the database with\n\n"
        "`docker compose -f deploy/docker-compose.dev.yml up -d`"
    )


st.title("Agent console")
st.caption("A development view of the system. Real where it says real.")

tab_tables, tab_agents, tab_chat = st.tabs(["1 · Tables", "2 · Agents", "3 · Ask"])


# --------------------------------------------------------------------------
# 1 - tables. REAL.
# --------------------------------------------------------------------------

with tab_tables:
    schema = st.session_state.schema

    if schema is None:
        st.info("Connect on the left to read the schema.")
    elif not schema:
        st.warning(
            f"`{user}` can read no tables in this database. That is a real answer, "
            "not an error - the role has no grants."
        )
    else:
        st.success(
            f"Read from the database. `{user}` can see "
            f"{len(schema)} table(s): {', '.join(sorted(schema))}."
        )
        st.caption(
            "Nothing here is configured anywhere. The columns, the types, and which "
            "columns are semantically searchable all come out of the catalogue."
        )

        chosen = st.selectbox("Table", sorted(schema))
        table = schema[chosen]

        rows = []
        for column in table.columns:
            partner = table.embedding_partner(column.name)
            rows.append({
                "column": column.name,
                "type": column.sql_type,
                "nullable": "yes" if column.nullable else "no",
                "embedding": partner.name if partner else "",
                "filter kind": classify_preview(table, column.name),
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        searchable = [c.name for c in table.columns if table.embedding_partner(c.name)]
        left, right = st.columns(2)
        left.metric("Columns", len(table.columns))
        right.metric("Semantically searchable", len(searchable))
        if searchable:
            st.caption(
                "Searchable because a matching `embed_<column>` vector column exists: "
                + ", ".join(f"`{n}`" for n in searchable)
            )

        st.divider()
        st.markdown("**Distinct values** — the number the ENUM rule turns on.")
        text_columns = [
            c.name for c in table.filterable_columns if c.sql_type == "text"
        ]
        if text_columns:
            column_name = st.selectbox("Column", text_columns, key="distinct_column")
            if st.button("Count"):
                try:
                    n = distinct_count(connection, chosen, column_name)
                    st.metric(f"{chosen}.{column_name}", f"{n} distinct")
                    st.caption(
                        "Few distinct values means the model can be given the list and "
                        "match what is actually stored. Many means it cannot."
                    )
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("No text columns in this table.")

        with st.expander("Try this"):
            st.markdown(
                "Reconnect as `app_catalog`, then as `app_circulation`, and watch the "
                "table list change.\n\n"
                "Neither role has a list of tables written down anywhere in the code. "
                "`information_schema` reports only what the connected role holds a "
                "privilege on, so the `GRANT` statements in `seeds/003_roles.sql` are "
                "the only place an agent's scope is defined."
            )


# --------------------------------------------------------------------------
# 2 - agents. Real table list, everything else a stub.
# --------------------------------------------------------------------------

with tab_agents:
    stub_note("agents live in this browser tab only. Feature 3 writes them to a registry.")

    schema = st.session_state.schema
    if schema is None:
        st.info("Connect first — an agent is defined by the tables it may read.")
    else:
        with st.form("agent"):
            col_a, col_b = st.columns(2)
            key = col_a.text_input("Key", placeholder="catalog",
                                   help="Lowercase, digits and underscores.")
            display_name = col_b.text_input("Name", placeholder="Catalogue")

            description = st.text_area(
                "Description",
                placeholder="Books, their authors and publishers…",
                help=(
                    "What the orchestrator reads when deciding whether to route here. "
                    "This one steers routing, so it matters more than the prompt."
                ),
                height=80,
            )
            prompt = st.text_area(
                "Prompt",
                placeholder="You answer questions about a library's catalogue…",
                help=(
                    "The agent's own instructions. Safe for a user to write: the GRANT "
                    "is the boundary, and no wording here can widen it."
                ),
                height=140,
            )
            tables = st.multiselect("Tables", sorted(schema),
                                    help="Read from the database, not typed by hand.")

            if st.form_submit_button("Save agent", type="primary"):
                if not key or not description or not tables:
                    st.error("Key, description and at least one table are required.")
                else:
                    st.session_state.agents = save_agent(
                        st.session_state.agents,
                        AgentDraft(
                            key=key.strip().lower(),
                            display_name=display_name or key,
                            description=description,
                            prompt=prompt,
                            tables=tables,
                            db_role=suggested_role_name(key.strip().lower()),
                        ),
                    )
                    st.success(f"Saved `{key}` in memory.")

    if st.session_state.agents:
        st.divider()
        st.markdown("**Defined so far**")
        for agent in st.session_state.agents:
            with st.expander(f"`{agent.key}` — {agent.display_name}"):
                st.write(agent.description)
                st.markdown(
                    f"Tables: {', '.join(f'`{t}`' for t in agent.tables)}  \n"
                    f"Would connect as: `{agent.db_role}`  \n"
                    f"Status: `{agent.status}`"
                )
                st.caption(
                    "Status is `pending` because no role exists yet. Provisioning "
                    "is a separate step with privileges the request path never holds, "
                    "and an agent is not routable until it completes."
                )


# --------------------------------------------------------------------------
# 3 - ask. STUB.
# --------------------------------------------------------------------------

with tab_chat:
    stub_note(
        "no model is called and no SQL runs. The trace shows the path a question "
        "will take once the orchestrator is wired up."
    )

    if not st.session_state.agents:
        st.info("Define an agent first — the orchestrator routes by description.")

    for entry in st.session_state.chat:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            for step in entry["result"]["steps"]:
                icon = {"route": "🧭", "delegate": "📨", "sql": "🗄️", "result": "📄"}
                st.markdown(f"{icon.get(step['kind'], '•')} {step['detail']}")
                if "payload" in step:
                    st.json(step["payload"], expanded=False)
            st.markdown(entry["result"]["answer"])

    question = st.chat_input("Ask something about the data…")
    if question:
        st.session_state.chat.append({
            "question": question,
            "result": ask(question, st.session_state.agents),
        })
        st.rerun()


# --------------------------------------------------------------------------
# what is real
# --------------------------------------------------------------------------

st.divider()
with st.expander("What in here is real"):
    st.markdown(
        """
| Screen | State | Becomes real with |
|---|---|---|
| Tables — list, columns, types, embedding partners | **real** | done |
| Tables — distinct counts | **real** | done |
| Tables — filter kind | preview | feature 2, the classifier |
| Agents — table choices | **real** | done |
| Agents — saving | stub, in memory | feature 3, the registry |
| Agents — Postgres role | stub, name only | phase 4, the provisioner |
| Ask — everything | stub | feature 4, then the orchestrator wiring |
| Login | absent | left until last, on purpose |
"""
    )
