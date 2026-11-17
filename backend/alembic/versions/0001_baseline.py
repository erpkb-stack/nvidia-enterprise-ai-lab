"""baseline: projects and architectures (v0.1 schema)

Revision ID: 0001
Revises:
Create Date: 2026-11-17
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(180), nullable=False),
        sa.Column("customer", sa.String(180)),
        sa.Column("use_case", sa.String(120)),
        sa.Column("status", sa.String(40)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_projects_id", "projects", ["id"])
    op.create_table(
        "architectures",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(180), nullable=False),
        sa.Column("blueprint", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_architectures_id", "architectures", ["id"])
    op.create_index("ix_architectures_project_id", "architectures", ["project_id"])


def downgrade() -> None:
    op.drop_table("architectures")
    op.drop_table("projects")
