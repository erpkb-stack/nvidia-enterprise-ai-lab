"""flagship modules: RAG documents/chunks/query log, benchmark runs, validation reports

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-17
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("collection", sa.String(64)),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100)),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column("pages", sa.Integer()),
        sa.Column("chunk_count", sa.Integer()),
        sa.Column("embedding_provider", sa.String(80)),
        sa.Column("status", sa.String(32)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_rag_documents_id", "rag_documents", ["id"])
    op.create_index("ix_rag_documents_collection", "rag_documents", ["collection"])
    op.create_index("ix_rag_documents_sha256", "rag_documents", ["sha256"])

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(),
                  sa.ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("collection", sa.String(64)),
        sa.Column("ordinal", sa.Integer()),
        sa.Column("page", sa.Integer()),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_rag_chunks_id", "rag_chunks", ["id"])
    op.create_index("ix_rag_chunks_document_id", "rag_chunks", ["document_id"])
    op.create_index("ix_rag_chunks_collection", "rag_chunks", ["collection"])

    op.create_table(
        "rag_queries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("collection", sa.String(64)),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text()),
        sa.Column("citations", sa.JSON()),
        sa.Column("providers", sa.JSON()),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_rag_queries_id", "rag_queries", ["id"])

    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.String(32)),
        sa.Column("label", sa.String(180)),
        sa.Column("params", sa.JSON()),
        sa.Column("results", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_benchmark_runs_id", "benchmark_runs", ["id"])

    op.create_table(
        "validation_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("customer", sa.String(180)),
        sa.Column("title", sa.String(255)),
        sa.Column("status", sa.String(32)),
        sa.Column("config", sa.JSON()),
        sa.Column("checks", sa.JSON()),
        sa.Column("markdown", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_validation_reports_id", "validation_reports", ["id"])


def downgrade() -> None:
    for t in ("validation_reports", "benchmark_runs", "rag_queries", "rag_chunks", "rag_documents"):
        op.drop_table(t)
