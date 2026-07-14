#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cases=(
  no_data_low_completed_sales_future_date
  clarification_missing_factual_question
  clarification_missing_staff_entity
  clarification_missing_date_range
  area_stores_count
  cash_deposit_total_month
  compensation_top_location_month
  compensation_top_paid_staff_month
  compensation_top_staff_month
  payments_deposits_01
  sales_store_count_this_month
  sales_store_top_this_month
  shift_count_today
  staff_top_today
  memory_location_sales_count_followup_replaces_location
  memory_temporal_sales_count_followup_replaces_day
  memory_subject_change_locations_to_stores_count
  staff_id_sales_count_today
  staff_id_pair_sales_count_today
  memory_repeated_named_target_reuses_canonical_identity
)

case_ids="$(IFS=,; printf '%s' "${cases[*]}")"

export FERVIS_GOLDSET_CASE_IDS="$case_ids"
export FERVIS_GOLDSET_SUITE="fervis_goldsets.ozana.suite:load_suite"
export FERVIS_GOLDSET_TENANT_ID="${FERVIS_GOLDSET_TENANT_ID:-local-goldset}"

exec "$repo_root/scripts/run-local-goldset.sh" "$@"
