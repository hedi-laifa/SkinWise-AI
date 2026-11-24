/**
 * ACNE04 severity classes (Hayashi grading) and their display metadata.
 * Shared by every component that renders a class name, colour or description.
 */

export const CLASS_ORDER = [
  "Mild (Level 0)",
  "Moderate (Level 1)",
  "Severe (Level 2)",
  "Very Severe (Level 3)",
];

export const SEVERITY_META = {
  "Mild (Level 0)": {
    level: "Grade 0",
    shortLabel: "Mild",
    lesionRange: "1–5 papules/pustules",
    barColor: "bg-emerald-500",
    ringColor: "#10b981",
    badgeCls: "bg-emerald-50 text-emerald-700 border-emerald-200",
    cardCls: "border-emerald-300 bg-emerald-50",
    dot: "bg-emerald-500",
    description:
      "Minimal comedones or sparse superficial inflammatory lesions localized to a single facial zone.",
    recommendation:
      "Routine gentle cleansing and topical OTC keratolytics (salicylic acid / benzoyl peroxide).",
  },
  "Moderate (Level 1)": {
    level: "Grade 1",
    shortLabel: "Moderate",
    lesionRange: "6–20 papules/pustules",
    barColor: "bg-amber-400",
    ringColor: "#f59e0b",
    badgeCls: "bg-amber-50 text-amber-700 border-amber-200",
    cardCls: "border-amber-300 bg-amber-50",
    dot: "bg-amber-400",
    description:
      "Moderate papules and pustules distributed across multiple facial zones without deep scarring.",
    recommendation:
      "Consider combination topical retinoid + antimicrobial therapy under dermatological guidance.",
  },
  "Severe (Level 2)": {
    level: "Grade 2",
    shortLabel: "Severe",
    lesionRange: "21–50 papules/pustules",
    barColor: "bg-orange-500",
    ringColor: "#f97316",
    badgeCls: "bg-orange-50 text-orange-700 border-orange-200",
    cardCls: "border-orange-300 bg-orange-50",
    dot: "bg-orange-500",
    description:
      "Extensive inflammatory papules, pustules, and early nodular lesions with elevated erythema.",
    recommendation:
      "Clinical dermatologist evaluation advised for systemic + topical prescription regimen.",
  },
  "Very Severe (Level 3)": {
    level: "Grade 3",
    shortLabel: "Very Severe",
    lesionRange: ">50 inflammatory lesions",
    barColor: "bg-rose-500",
    ringColor: "#f43f5e",
    badgeCls: "bg-rose-50 text-rose-700 border-rose-200",
    cardCls: "border-rose-300 bg-rose-50",
    dot: "bg-rose-500",
    description:
      "High lesion density with deep nodulocystic involvement and high risk of permanent scarring.",
    recommendation:
      "Prompt specialist dermatology referral recommended for targeted systemic intervention.",
  },
};

export const DEFAULT_META = SEVERITY_META["Mild (Level 0)"];
