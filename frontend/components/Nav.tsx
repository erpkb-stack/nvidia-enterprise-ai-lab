"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const GROUPS: { title: string; links: [string, string, boolean?][] }[] = [
  { title: "OVERVIEW", links: [["Dashboard", "/"]] },
  {
    title: "FLAGSHIP LABS",
    links: [
      ["Enterprise RAG Assistant", "/rag", true],
      ["LLM Inference Performance Lab", "/benchmark", true],
      ["Solution Architect Studio", "/architect", true],
      ["GPU & Kubernetes Ops Lab", "/mlops", true],
      ["OEM Validation Center", "/oem", true],
    ],
  },
  {
    title: "REFERENCE",
    links: [
      ["Technology Explorer", "/technologies"],
      ["Model Serving", "/serving"],
      ["Fine-Tuning", "/finetuning"],
      ["AI Agent Studio", "/agents"],
      ["Documentation Center", "/docs-center"],
    ],
  },
];

export default function Nav() {
  const path = usePathname();
  return (
    <aside className="side">
      <Link href="/" className="brand">
        <span className="mark">N</span>NVIDIA AI LAB
      </Link>
      <nav className="nav" aria-label="Modules">
        {GROUPS.map((g) => (
          <div key={g.title}>
            <div className="group">{g.title}</div>
            {g.links.map(([label, href, flagship]) => (
              <Link key={href} href={href} className={path === href ? "active" : ""}>
                {label}
                {flagship && <span className="dot" aria-hidden />}
              </Link>
            ))}
          </div>
        ))}
      </nav>
      <p className="sub sidefoot">
        ENTERPRISE AI SOLUTIONS ARCHITECT LAB
        <br />
        v1.0.0
      </p>
    </aside>
  );
}
