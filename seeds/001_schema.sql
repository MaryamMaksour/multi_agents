-- Development schema: a small public-library catalogue.
--
-- This exists so the agents can be developed and tested against a real
-- database without pointing at anyone's production data. It is deliberately
-- unrelated to any deployment's real domain: six tables, a few thousand rows,
-- and it runs in about a second.
--
-- It is not a fixture chosen at random - every retrieval path the SQL tool
-- has to support is represented at least once:
--
--   numeric comparison   page_count, publication_year, price, fine_amount
--   low-cardinality text language, genre, membership_tier, status, city
--   real timestamps      borrowed_at, due_at, returned_at, joined_at, opened_on
--                        (proper timestamptz/date - NOT dates stored as text)
--   semantic search      summary, bio, and the title/name columns, each paired
--                        with an embed_<column> vector column
--   bilingual columns    every human-facing name exists as _en and _ar
--   pagination           loans is large enough to need more than one page
--   aggregation          counts and sums per branch, per genre, per tier
--   cross-agent join     loans -> books spans the two agent domains below
--
-- The two domains are meant to be served by two different agents:
--
--   catalog       authors, publishers, books
--   circulation   branches, members, loans
--
-- Neither list is written into application code. An agent's readable tables
-- come from its GRANTs (see 003_roles.sql), and introspection through that
-- role reports exactly those tables and nothing else.
--
-- Embedding dimension: 1024, matching BGE-M3 and multilingual-e5-large. If
-- you serve a model with a different dimension, change it here - it appears
-- only in this file - and re-run.
--
-- Usage:
--   createdb library_dev
--   psql -d library_dev -f seeds/001_schema.sql
--   python seeds/002_generate_data.py > seeds/002_data.sql
--   psql -d library_dev -f seeds/002_data.sql
--   psql -d library_dev -f seeds/003_roles.sql

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS loans, members, branches, books, publishers, authors CASCADE;

-- ============================================================
-- catalog domain
-- ============================================================

CREATE TABLE authors (
    id            integer PRIMARY KEY,
    name_en       text NOT NULL,
    name_ar       text,
    nationality   text,
    birth_year    integer,
    bio           text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    embed_name_en vector(1024),
    embed_name_ar vector(1024),
    embed_bio     vector(1024)
);

CREATE TABLE publishers (
    id            integer PRIMARY KEY,
    name_en       text NOT NULL,
    name_ar       text,
    city          text,
    country       text,
    founded_year  integer,
    embed_name_en vector(1024),
    embed_name_ar vector(1024)
);

CREATE TABLE books (
    id               integer PRIMARY KEY,
    title_en         text NOT NULL,
    title_ar         text,
    author_id        integer REFERENCES authors(id),
    publisher_id     integer REFERENCES publishers(id),
    isbn             text UNIQUE,
    publication_year integer,
    page_count       integer,
    language         text,
    genre            text,
    summary          text,
    shelf_code       text,
    copies_total     integer NOT NULL DEFAULT 1,
    price            numeric(10,2),
    added_at         timestamptz NOT NULL DEFAULT now(),
    embed_title_en   vector(1024),
    embed_title_ar   vector(1024),
    embed_summary    vector(1024)
);

-- ============================================================
-- circulation domain
-- ============================================================

CREATE TABLE branches (
    id            integer PRIMARY KEY,
    name_en       text NOT NULL,
    name_ar       text,
    city          text,
    address       text,
    opened_on     date,
    embed_name_en vector(1024),
    embed_address vector(1024)
);

CREATE TABLE members (
    id                 integer PRIMARY KEY,
    full_name_en       text NOT NULL,
    full_name_ar       text,
    email              text UNIQUE,
    phone              text,
    membership_tier    text,
    status             text,
    city               text,
    joined_at          timestamptz NOT NULL,
    embed_full_name_en vector(1024),
    embed_full_name_ar vector(1024)
);

CREATE TABLE loans (
    id          integer PRIMARY KEY,
    book_id     integer NOT NULL REFERENCES books(id),
    member_id   integer NOT NULL REFERENCES members(id),
    branch_id   integer NOT NULL REFERENCES branches(id),
    borrowed_at timestamptz NOT NULL,
    due_at      timestamptz NOT NULL,
    returned_at timestamptz,
    status      text NOT NULL,
    fine_amount numeric(8,2) NOT NULL DEFAULT 0
);

-- ============================================================
-- indexes
-- ============================================================

CREATE INDEX idx_books_author      ON books(author_id);
CREATE INDEX idx_books_publisher   ON books(publisher_id);
CREATE INDEX idx_books_genre       ON books(genre);
CREATE INDEX idx_books_year        ON books(publication_year);

CREATE INDEX idx_loans_book        ON loans(book_id);
CREATE INDEX idx_loans_member      ON loans(member_id);
CREATE INDEX idx_loans_branch      ON loans(branch_id);
CREATE INDEX idx_loans_status      ON loans(status);
CREATE INDEX idx_loans_borrowed_at ON loans(borrowed_at);

CREATE INDEX idx_members_tier      ON members(membership_tier);
CREATE INDEX idx_members_status    ON members(status);

-- Vector indexes are created but stay empty until an embedding backfill runs;
-- an HNSW index over all-NULL columns is valid and costs nothing. At this row
-- count an exact scan is often faster anyway - these exist so the query plans
-- you develop against match the ones you get with real volume.
CREATE INDEX idx_books_embed_summary  ON books   USING hnsw (embed_summary  vector_cosine_ops);
CREATE INDEX idx_books_embed_title_en ON books   USING hnsw (embed_title_en vector_cosine_ops);
CREATE INDEX idx_books_embed_title_ar ON books   USING hnsw (embed_title_ar vector_cosine_ops);
CREATE INDEX idx_authors_embed_bio    ON authors USING hnsw (embed_bio      vector_cosine_ops);
