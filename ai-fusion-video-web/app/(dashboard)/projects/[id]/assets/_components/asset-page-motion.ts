import type { Variants } from "framer-motion";

export const assetPageVariants: Variants = {
  visible: { transition: { staggerChildren: 0.04 } },
};

export const assetSectionVariants: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};
