import React from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import type { HTMLAttributes, ReactNode } from "react";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface MarkdownViewerProps {
  content: string;
}

interface CodeProps extends HTMLAttributes<HTMLElement> {
  inline?: boolean;
  children?: ReactNode;
}

const CodeBlock: React.FC<CodeProps> = ({ inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className ?? "");
  const code = String(children ?? "").replace(/\n$/, "");

  if (!inline && match) {
    return (
      <SyntaxHighlighter PreTag="div" language={match[1]} style={oneDark} wrapLongLines>
        {code}
      </SyntaxHighlighter>
    );
  }

  return (
    <code className={className} {...props}>
      {children}
    </code>
  );
};

const markdownComponents: Components = {
  code: CodeBlock
};

const MarkdownViewer: React.FC<MarkdownViewerProps> = ({ content }) => (
  <div className="markdown-viewer">
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
      {content}
    </ReactMarkdown>
  </div>
);

export default MarkdownViewer;
