#!/usr/bin/env python3
"""
技能系统渐进式加载算法演示

这个脚本演示了渐进式加载的核心流程：
1. 扫描技能目录，解析 YAML frontmatter，提取元数据（渐进式加载）
2. 将元数据注入系统提示，让 Agent 看到技能列表
3. Agent 识别任务并选择适用的技能
4. Agent 按需读取完整技能内容
5. Agent 按照技能指导执行任务
"""

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, TypedDict

import dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    try:
        from pydantic import BaseModel, Field
        from pydantic import ValidationError
    except ImportError:
        from pydantic import BaseModel, Field
        from pydantic import ValidationError

# 加载环境变量
dotenv.load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


# ============================================================================
# 类型定义
# ============================================================================

class SkillMetadata(TypedDict):
    """技能元数据"""
    name: str
    description: str
    path: str
    source: str


# ============================================================================
# 核心算法：技能加载（从 load.py 复现）
# ============================================================================

MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """检查路径是否安全"""
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """解析 SKILL.md 文件的 YAML frontmatter（渐进式加载：只解析元数据）"""
    try:
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            return None

        content = skill_md_path.read_text(encoding="utf-8")
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return None

        frontmatter = match.group(1)
        metadata: dict[str, str] = {}
        for line in frontmatter.split("\n"):
            kv_match = re.match(r"^(\w+):\s*(.+)$", line.strip())
            if kv_match:
                key, value = kv_match.groups()
                metadata[key] = value.strip()

        if "name" not in metadata or "description" not in metadata:
            return None

        return SkillMetadata(
            name=metadata["name"],
            description=metadata["description"],
            path=str(skill_md_path),
            source=source,
        )

    except (OSError, UnicodeDecodeError):
        return None


def _list_skills(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """扫描单个技能目录"""
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        return []

    skills: list[SkillMetadata] = []

    for skill_dir in skills_dir.iterdir():
        if not _is_safe_path(skill_dir, resolved_base):
            continue
        if not skill_dir.is_dir():
            continue

        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(
    *, user_skills_dir: Path | None = None, project_skills_dir: Path | None = None
) -> list[SkillMetadata]:
    """合并用户级和项目级技能"""
    all_skills: dict[str, SkillMetadata] = {}

    if user_skills_dir:
        user_skills = _list_skills(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    if project_skills_dir:
        project_skills = _list_skills(project_skills_dir, source="project")
        for skill in project_skills:
            all_skills[skill["name"]] = skill

    return list(all_skills.values())


# ============================================================================
# 系统提示构建（从 middleware.py 复现）
# ============================================================================

SKILLS_SYSTEM_PROMPT = """
## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you know they exist (name + description above), but you only read the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches any skill's description
2. **Read the skill's full instructions**: The skill list above shows the exact path to use with read_file
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When you need specialized knowledge or structured workflows
- When a skill provides proven patterns for complex tasks

**Skills are Self-Documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it
- The skill list above shows the full path for each skill's SKILL.md file
"""


def format_skills_locations(user_skills_display: str, project_skills_dir: Path | None = None) -> str:
    """格式化技能位置信息"""
    locations = [f"**User Skills**: `{user_skills_display}`"]
    if project_skills_dir:
        locations.append(f"**Project Skills**: `{project_skills_dir}` (overrides user skills)")
    return "\n".join(locations)


def format_skills_list(skills: list[SkillMetadata], user_skills_display: str, project_skills_dir: Path | None = None) -> str:
    """格式化技能列表（渐进式加载：只显示元数据）"""
    if not skills:
        locations = [f"{user_skills_display}/"]
        if project_skills_dir:
            locations.append(f"{project_skills_dir}/")
        return f"(No skills available yet. You can create skills in {' or '.join(locations)})"

    user_skills = [s for s in skills if s["source"] == "user"]
    project_skills = [s for s in skills if s["source"] == "project"]

    lines = []

    if user_skills:
        lines.append("**User Skills:**")
        for skill in user_skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  → Read `{skill['path']}` for full instructions")
        lines.append("")

    if project_skills:
        lines.append("**Project Skills:**")
        for skill in project_skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  → Read `{skill['path']}` for full instructions")

    return "\n".join(lines)


def build_skills_system_prompt(
    skills: list[SkillMetadata],
    user_skills_display: str,
    project_skills_dir: Path | None = None,
) -> str:
    """构建包含技能信息的系统提示（渐进式加载：只包含元数据）"""
    skills_locations = format_skills_locations(user_skills_display, project_skills_dir)
    skills_list = format_skills_list(skills, user_skills_display, project_skills_dir)

    return SKILLS_SYSTEM_PROMPT.format(
        skills_locations=skills_locations,
        skills_list=skills_list,
    )


# ============================================================================
# 大模型创建（从 config.py 复现）
# ============================================================================

# ============================================================================
# 结构化输出：技能选择
# ============================================================================

class SkillSelection(BaseModel):
    """技能选择的结构化输出"""
    skill_name: str = Field(description="要使用的技能名称（必须与可用技能列表中的名称完全匹配）")
    reason: str = Field(description="选择该技能的原因")
    confidence: float = Field(description="选择该技能的置信度（0-1）", ge=0, le=1)


class ScriptAction(BaseModel):
    """脚本操作的结构化输出"""
    action: Literal["read_script", "execute_function", "continue"] = Field(
        description="要执行的操作：read_script=读取脚本文件, execute_function=执行函数, continue=继续对话"
    )
    script_name: str | None = Field(
        default=None,
        description="脚本文件名（如 web_search.py），当 action=read_script 或 execute_function 时必需"
    )
    function_name: str | None = Field(
        default=None,
        description="要调用的函数名（如 web_search），当 action=execute_function 时必需"
    )
    function_params: dict[str, Any] | None = Field(
        default=None,
        description="函数参数字典（如 {{'query': 'quantum computing', 'max_results': 5}}），当 action=execute_function 时必需"
    )
    reasoning: str = Field(description="执行此操作的原因说明")


# ============================================================================
# 脚本执行器：动态加载和执行技能脚本
# ============================================================================

def execute_skill_script(script_path: Path, function_name: str, **kwargs) -> Any:
    """动态加载并执行技能脚本中的函数
    
    Args:
        script_path: Python 脚本文件的路径
        function_name: 要调用的函数名
        **kwargs: 传递给函数的参数
    
    Returns:
        函数执行结果
    """
    if not script_path.exists():
        return {"error": f"脚本文件不存在: {script_path}"}
    
    try:
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("skill_script", script_path)
        if spec is None or spec.loader is None:
            return {"error": f"无法加载脚本: {script_path}"}
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["skill_script"] = module
        spec.loader.exec_module(module)
        
        # 获取函数
        if not hasattr(module, function_name):
            available_functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith("_")]
            return {
                "error": f"函数 '{function_name}' 不存在",
                "available_functions": available_functions
            }
        
        func = getattr(module, function_name)
        
        # 执行函数
        result = func(**kwargs)
        return result
        
    except Exception as e:
        return {"error": f"执行脚本时出错: {str(e)}"}


def list_skill_scripts(skill_dir: Path) -> list[str]:
    """列出技能目录中的所有 Python 脚本文件"""
    scripts = []
    if not skill_dir.exists():
        return scripts
    
    for file in skill_dir.iterdir():
        if file.is_file() and file.suffix == ".py" and file.name != "__init__.py":
            scripts.append(file.name)
    
    return scripts


# ============================================================================
# 大模型创建
# ============================================================================

def create_model() -> BaseChatModel:
    """创建大模型实例"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            print(f"🤖 使用 OpenAI 模型: {model_name}")
            return ChatOpenAI(model=model_name, temperature=0)
        except Exception as e:
            print(f"⚠️  OpenAI 模型加载失败: {e}")
            print("   尝试其他模型...")

    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
            print(f"🤖 使用 Anthropic 模型: {model_name}")
            return ChatAnthropic(model_name=model_name, max_tokens=20_000)  # type: ignore
        except Exception as e:
            print(f"⚠️  Anthropic 模型加载失败: {e}")
            print("   尝试其他模型...")

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = os.environ.get("GOOGLE_MODEL", "gemini-3-pro-preview")
            print(f"🤖 使用 Google Gemini 模型: {model_name}")
            return ChatGoogleGenerativeAI(model=model_name, temperature=0)
        except Exception as e:
            print(f"⚠️  Google Gemini 模型加载失败: {e}")

    raise ValueError("未找到可用的 API 密钥或模型加载失败。请设置 OPENAI_API_KEY、ANTHROPIC_API_KEY 或 GOOGLE_API_KEY")


# ============================================================================
# 测试场景：初始化技能文件
# ============================================================================

def get_skills_directories() -> tuple[Path, Path]:
    """获取技能目录路径"""
    script_dir = Path(__file__).parent
    user_skills_dir = script_dir / "skills" / "user-skills"
    project_skills_dir = script_dir / "skills" / "project-skills"
    return user_skills_dir, project_skills_dir


def init_test_skills_if_needed(user_skills_dir: Path, project_skills_dir: Path):
    """如果技能目录为空，则初始化示例技能文件"""
    # 检查用户技能目录
    user_has_skills = any(user_skills_dir.iterdir()) if user_skills_dir.exists() else False
    
    if not user_has_skills:
        print(f"\n📝 初始化用户技能目录: {user_skills_dir}")
        user_skills_dir.mkdir(parents=True, exist_ok=True)
        
        # 技能1：web-research
        web_research_dir = user_skills_dir / "web-research"
        web_research_dir.mkdir(exist_ok=True)
        (web_research_dir / "SKILL.md").write_text("""---
name: web-research
description: Structured approach to conducting thorough web research
---

# Web Research Skill

This skill provides a structured workflow for conducting comprehensive web research.

## When to Use
- User asks you to research a topic
- Need to gather information from multiple sources
- Want to synthesize information from web content

## Workflow
1. **Define research objectives**: Clearly identify what information you need to find
2. **Search multiple sources**: Use web_search tool to query different aspects
3. **Evaluate credibility**: Check source reliability and recency
4. **Synthesize findings**: Combine information from multiple sources into coherent insights
5. **Present results**: Organize findings with clear structure and citations

## Best Practices
- Always search from multiple angles
- Verify information across sources
- Focus on recent and authoritative sources
- Provide citations for all claims
""", encoding="utf-8")

        # 技能2：code-review
        code_review_dir = user_skills_dir / "code-review"
        code_review_dir.mkdir(exist_ok=True)
        (code_review_dir / "SKILL.md").write_text("""---
name: code-review
description: Systematic code review checklist and best practices
---

# Code Review Skill

This skill provides a comprehensive checklist for code reviews.

## Review Checklist
- [ ] **Code Style**: Follows project style guide and conventions
- [ ] **Security**: No security vulnerabilities (SQL injection, XSS, etc.)
- [ ] **Error Handling**: Proper error handling and edge cases covered
- [ ] **Testing**: Tests included and passing
- [ ] **Documentation**: Code is well-documented
- [ ] **Performance**: No obvious performance issues
- [ ] **Dependencies**: Dependencies are necessary and up-to-date

## Review Process
1. Read the code carefully
2. Check each item in the checklist
3. Provide constructive feedback
4. Suggest improvements where applicable
""", encoding="utf-8")
        print("  ✓ 创建了 2 个用户技能")

    # 检查项目技能目录
    project_has_skills = any(project_skills_dir.iterdir()) if project_skills_dir.exists() else False
    
    if not project_has_skills:
        print(f"\n📝 初始化项目技能目录: {project_skills_dir}")
        project_skills_dir.mkdir(parents=True, exist_ok=True)

        # 项目特定的 web-research（覆盖用户技能）
        project_web_research_dir = project_skills_dir / "web-research"
        project_web_research_dir.mkdir(exist_ok=True)
        (project_web_research_dir / "SKILL.md").write_text("""---
name: web-research
description: Project-specific web research workflow with internal tools
---

# Web Research Skill (Project-Specific)

This is a project-specific version that overrides the user skill.

## Project-Specific Workflow
1. **Check internal knowledge base first**: Search project wiki and documentation
2. **Use project-specific search tools**: Leverage internal search APIs
3. **Follow project documentation standards**: Use project-specific citation format
4. **Submit findings to project wiki**: All research must be documented in project wiki

## Project Requirements
- All research must be peer-reviewed before publication
- Use project-specific templates for research reports
- Include project tags and categories
""", encoding="utf-8")

        # 项目特定技能
        project_specific_dir = project_skills_dir / "project-deployment"
        project_specific_dir.mkdir(exist_ok=True)
        (project_specific_dir / "SKILL.md").write_text("""---
name: project-deployment
description: Deployment procedures specific to this project
---

# Project Deployment Skill

This skill is only available at the project level.

## Deployment Steps
1. Run pre-deployment tests: `npm run test:pre-deploy`
2. Build Docker image: `docker build -t project:latest .`
3. Deploy to staging: `kubectl apply -f k8s/staging/`
4. Run smoke tests: `npm run test:smoke`
5. Deploy to production: `kubectl apply -f k8s/production/`

## Rollback Procedure
If deployment fails, run: `kubectl rollout undo deployment/project`
""", encoding="utf-8")
        print("  ✓ 创建了 2 个项目技能")




# ============================================================================
# 主测试函数
# ============================================================================

def test_progressive_loading_with_llm():
    """演示渐进式加载算法"""
    print("=" * 80)
    print("DeepAgents CLI 技能系统 - 渐进式加载算法演示")
    print("=" * 80)

    # 使用真实技能目录路径
    user_skills_dir, project_skills_dir = get_skills_directories()
    
    # 如果目录为空，初始化示例技能
    init_test_skills_if_needed(user_skills_dir, project_skills_dir)
    
    print(f"\n📁 使用技能目录:")
    print(f"  用户技能: {user_skills_dir}")
    print(f"  项目技能: {project_skills_dir}")

    print("\n" + "=" * 80)
    print("阶段 1: 扫描技能目录并提取元数据（渐进式加载）")
    print("=" * 80)

    # 阶段1：加载技能元数据
    skills = list_skills(
        user_skills_dir=user_skills_dir,
        project_skills_dir=project_skills_dir,
    )

    print(f"\n✅ 加载完成！共发现 {len(skills)} 个技能")
    print("\n技能元数据（仅名称和描述，不包含完整内容）:")
    for skill in skills:
        print(f"  - {skill['name']} ({skill['source']})")
        print(f"    描述: {skill['description']}")
        print()

    # 计算 token 节省
    total_skill_size = sum(Path(skill["path"]).stat().st_size for skill in skills)
    metadata_size = len(json.dumps([s for s in skills], indent=2))
    print(f"\n📊 Token 使用分析:")
    print(f"  完整技能内容大小: {total_skill_size:,} 字节")
    print(f"  元数据大小: {metadata_size:,} 字节")
    print(f"  节省: {total_skill_size - metadata_size:,} 字节 ({100 * (1 - metadata_size/total_skill_size):.1f}%)")

    print("\n" + "=" * 80)
    print("阶段 2: 构建系统提示（包含技能元数据）")
    print("=" * 80)

    # 阶段2：构建系统提示
    user_skills_display = str(user_skills_dir)
    skills_prompt = build_skills_system_prompt(
        skills=skills,
        user_skills_display=user_skills_display,
        project_skills_dir=project_skills_dir,
    )

    base_system_prompt = """You are a helpful AI assistant with access to a skills library.
Your role is to help users complete tasks by leveraging available skills when appropriate."""   

    full_system_prompt = base_system_prompt + "\n\n" + skills_prompt

    print("\n系统提示（包含技能列表）:")
    print("-" * 80)
    print(full_system_prompt)
    print("-" * 80)

    print("\n" + "=" * 80)
    print("阶段 3: Agent 识别并选择技能")
    print("=" * 80)

    # 创建模型
    try:
        model = create_model()
    except (ValueError, Exception) as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保在 .env 文件中配置了以下任一 API 密钥:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
        return

    # Agent 结构化选择技能
    print("\n📝 Agent 结构化选择技能")
    print("-" * 80)
    user_query1 = "我需要研究一下量子计算的最新进展，你能帮我吗？"
    print(f"用户查询: {user_query1}\n")

    # 构建技能选择提示（引用系统提示中已有的技能列表）
    skill_selection_prompt = f"""根据用户的任务，从系统提示中列出的可用技能中选择最合适的技能。

用户任务：{user_query1}

请以 JSON 格式返回你的选择，格式如下：
{{
    "skill_name": "技能名称（必须与系统提示中列出的技能名称完全匹配）",
    "reason": "选择该技能的原因",
    "confidence": 置信度分数（0-1）
}}

只返回 JSON，不要包含其他文字。"""

    selected_skill = None
    skill_selection_result = None

    try:
        # 使用结构化输出
        if hasattr(model, 'with_structured_output'):
            # OpenAI 和其他支持结构化输出的模型
            structured_model = model.with_structured_output(SkillSelection)
            print("🤖 Agent 思考中（结构化输出）...")
            skill_selection_result = structured_model.invoke([
                SystemMessage(content=full_system_prompt),
                HumanMessage(content=skill_selection_prompt),
            ])
            skill_selection_result = skill_selection_result.dict()
        else:
            # 回退到 JSON 解析
            messages1 = [
                SystemMessage(content=full_system_prompt + "\n\n重要：请以 JSON 格式返回技能选择。"),
                HumanMessage(content=skill_selection_prompt),
            ]
            print("🤖 Agent 思考中...")
            response1 = model.invoke(messages1)
            print(f"\nAgent 原始响应:\n{response1.content}\n")
            
            # 尝试从响应中提取 JSON
            json_match = re.search(r'\{[^{}]*"skill_name"[^{}]*\}', response1.content, re.DOTALL)
            if json_match:
                skill_selection_result = json.loads(json_match.group())
            else:
                # 尝试解析整个响应
                skill_selection_result = json.loads(response1.content.strip())
        
        print(f"\nAgent 技能选择:\n{json.dumps(skill_selection_result, indent=2, ensure_ascii=False)}\n")
    except (json.JSONDecodeError, ValidationError, AttributeError, NameError) as e:
        print(f"⚠️  解析技能选择失败: {e}")
        print("   尝试手动匹配...")
        # 回退：从响应中查找技能名称
        response1_content = ""
        try:
            if 'response1' in locals():
                response1_content = response1.content if hasattr(response1, 'content') else str(response1)
        except NameError:
            pass
        
        if response1_content:
            for skill in skills:
                if skill["name"].lower() in response1_content.lower():
                    skill_selection_result = {
                        "skill_name": skill["name"],
                        "reason": "从响应中自动识别",
                        "confidence": 0.7
                    }
                    break

    # 验证并找到对应的技能
    if skill_selection_result and "skill_name" in skill_selection_result:
        skill_name = skill_selection_result["skill_name"]
        print(f"✅ Agent 选择了技能: {skill_name}")
        print(f"   原因: {skill_selection_result.get('reason', 'N/A')}")
        print(f"   置信度: {skill_selection_result.get('confidence', 'N/A')}")
        
        # 动态查找技能
        selected_skill = next((s for s in skills if s["name"] == skill_name), None)
        if not selected_skill:
            print(f"⚠️  技能 '{skill_name}' 未找到，尝试模糊匹配...")
            # 模糊匹配
            for skill in skills:
                if skill_name.lower() in skill["name"].lower() or skill["name"].lower() in skill_name.lower():
                    selected_skill = skill
                    print(f"   找到匹配技能: {skill['name']}")
                    break
    else:
        print("⚠️  未能解析技能选择，使用默认技能...")
        # 默认使用 web-research
        selected_skill = next((s for s in skills if s["name"] == "web-research"), None)

    # Agent 按需读取完整技能内容（动态）
    print("\n" + "=" * 80)
    print("阶段 4: Agent 按需读取完整技能内容（动态解析）")
    print("=" * 80)

    if selected_skill:
        print(f"\n📖 Agent 读取完整技能内容: {selected_skill['name']}")
        print(f"   来源: {selected_skill['source']}")
        print(f"   路径: {selected_skill['path']}\n")

        full_skill_content = Path(selected_skill["path"]).read_text(encoding="utf-8")
        print("完整技能内容:")
        print("-" * 80)
        print(full_skill_content)
        print("-" * 80)

        # 获取技能目录路径
        skill_dir = Path(selected_skill["path"]).parent
        skill_scripts = list_skill_scripts(skill_dir)
        
        if skill_scripts:
            print(f"\n📜 发现技能脚本: {', '.join(skill_scripts)}")

        # Agent 使用技能执行任务
        print("\n📝 Agent 使用技能执行任务")
        print("-" * 80)

        # 构建包含完整技能内容的提示
        skill_usage_prompt = f"""基于以下技能指导，请帮助用户完成研究任务。

技能内容:
{full_skill_content}

用户任务: {user_query1}

**重要说明：**
1. 技能内容中可能提到了脚本文件（如 web_search.py），这些脚本位于技能目录中
2. 你需要先读取这些脚本文件，了解可用的函数和参数
3. 然后根据技能的工作流程，调用相应的函数来完成任务
4. 技能目录路径: {skill_dir}

**工作流程：**
1. 仔细阅读技能内容，识别需要使用的脚本文件
2. 使用 read_file 工具读取脚本文件，了解函数签名和用法
3. 根据技能指导，调用脚本中的函数执行任务
4. 将结果整合并按照技能要求呈现给用户

请开始执行任务。"""

        # 实际执行：让 Agent 自己发现和执行脚本
        print("🤖 Agent 按照技能指导执行任务...")
        
        # 构建系统提示，告诉 Agent 如何执行脚本
        script_execution_instructions = f"""
**脚本执行说明：**

技能目录中可能包含 Python 脚本文件（如 web_search.py）。要使用这些脚本：

1. **读取脚本文件**：当你需要了解脚本中的函数时，请明确说明要读取哪个脚本文件
2. **执行脚本函数**：当你需要调用脚本函数时，请明确说明：
   - 脚本文件名（如：web_search.py）
   - 函数名（如：web_search）
   - 函数参数（如：query="quantum computing", max_results=5）

系统会自动读取脚本文件或执行脚本函数并返回结果。

技能目录: {skill_dir}
可用脚本: {', '.join(skill_scripts) if skill_scripts else '无'}
"""
        
        messages2 = [
            SystemMessage(content=base_system_prompt + "\n\n" + script_execution_instructions),
            HumanMessage(content=skill_usage_prompt),
        ]
        
        # 交互循环：让 Agent 可以多次读取脚本和执行函数
        max_iterations = 10
        conversation_history = []
        loaded_scripts = {}  # 缓存已加载的脚本内容
        
        for iteration in range(max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            response2 = model.invoke(messages2 + conversation_history)
            print(f"\nAgent 响应:\n{response2.content}\n")
            
            if iteration == 0:
                conversation_history.append(HumanMessage(content=skill_usage_prompt))
            conversation_history.append(response2)
            
            # 检查 Agent 是否请求读取脚本或执行函数
            response_text = response2.content.lower()
            action_taken = False
            
            # 检查是否请求读取脚本文件
            for script_name in skill_scripts:
                script_name_lower = script_name.lower()
                # 检查是否明确提到要读取脚本
                if script_name_lower in response_text and (
                    "read" in response_text or "读取" in response_text or 
                    "查看" in response_text or "查看" in response_text or
                    "了解" in response_text or "看看" in response_text
                ):
                    script_path = skill_dir / script_name
                    if script_name not in loaded_scripts:
                        print(f"\n📖 Agent 请求读取脚本: {script_name}")
                        script_content = script_path.read_text(encoding="utf-8")
                        loaded_scripts[script_name] = script_content
                        print(f"脚本内容:\n{script_content}\n")
                        
                        conversation_history.append(HumanMessage(
                            content=f"脚本文件 {script_name} 的内容:\n\n```python\n{script_content}\n```\n\n请根据脚本中的函数定义，说明你要调用哪个函数以及传递什么参数。"
                        ))
                        action_taken = True
                        break
            
            # 检查是否请求执行脚本函数
            if not action_taken:
                for script_name in skill_scripts:
                    script_name_lower = script_name.lower()
                    if script_name_lower in response_text:
                        # 先确保脚本已加载
                        if script_name not in loaded_scripts:
                            script_path = skill_dir / script_name
                            script_content = script_path.read_text(encoding="utf-8")
                            loaded_scripts[script_name] = script_content
                            print(f"\n📖 自动加载脚本: {script_name}")
                        
                        # 从脚本内容中提取函数名
                        script_content = loaded_scripts[script_name]
                        # 查找函数定义
                        function_pattern = r'def\s+(\w+)\s*\('
                        functions = re.findall(function_pattern, script_content)
                        
                        if functions:
                            # 尝试识别要调用的函数
                            function_name = None
                            for func in functions:
                                if func.lower() in response_text:
                                    function_name = func
                                    break
                            
                            # 如果没有明确提到，使用第一个函数
                            if not function_name and functions:
                                function_name = functions[0]
                            
                            if function_name:
                                # 尝试提取参数
                                params = {}
                                
                                # 从用户查询中提取搜索关键词
                                if "quantum" in user_query1.lower() or "量子" in user_query1:
                                    params["query"] = "quantum computing latest advances 2024"
                                
                                # 从响应中提取参数
                                query_match = re.search(r'query[=:：]\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                                if query_match:
                                    params["query"] = query_match.group(1)
                                
                                # 查找 max_results
                                max_results_match = re.search(r'max_results[=:：]\s*(\d+)', response_text, re.IGNORECASE)
                                if max_results_match:
                                    params["max_results"] = int(max_results_match.group(1))
                                elif "max_results" not in params:
                                    params["max_results"] = 5
                                
                                # 如果函数需要 query 参数但没有提取到，从上下文推断
                                if "query" not in params and "web_search" in function_name.lower():
                                    # 从用户查询中提取关键词
                                    if "量子计算" in user_query1:
                                        params["query"] = "quantum computing latest advances 2024"
                                    else:
                                        # 尝试从响应中提取
                                        words = user_query1.split()
                                        if words:
                                            params["query"] = " ".join(words[-5:])  # 使用最后几个词
                                
                                print(f"\n⚙️  执行脚本函数: {script_name} -> {function_name}")
                                print(f"   参数: {params}")
                                
                                script_path = skill_dir / script_name
                                result = execute_skill_script(script_path, function_name, **params)
                                
                                if "error" in result:
                                    print(f"  ❌ 执行失败: {result['error']}")
                                    conversation_history.append(HumanMessage(
                                        content=f"执行脚本函数失败: {result['error']}\n请检查函数名和参数是否正确。"
                                    ))
                                else:
                                    print(f"  ✅ 执行成功")
                                    # 格式化结果
                                    if isinstance(result, dict) and "results" in result:
                                        results_text = "\n".join([
                                            f"- **{r.get('title', 'N/A')}**\n  URL: {r.get('url', 'N/A')}\n  {r.get('content', '')[:300]}..."
                                            for r in result.get('results', [])[:5]
                                        ])
                                        conversation_history.append(HumanMessage(
                                            content=f"脚本函数执行结果:\n\n{results_text}\n\n请基于这些搜索结果，按照技能指导的工作流程，为用户生成完整的研究报告。"
                                        ))
                                    else:
                                        conversation_history.append(HumanMessage(
                                            content=f"脚本函数执行结果:\n\n{json.dumps(result, indent=2, ensure_ascii=False)}\n\n请基于这些结果完成任务。"
                                        ))
                                action_taken = True
                                break
            
            # 如果 Agent 没有请求执行脚本，检查是否已经完成任务
            if not action_taken:
                # 检查是否已经提供了最终答案
                if len(response2.content) > 500 and any(keyword in response_text for keyword in ["完成", "总结", "报告", "report", "summary", "结论"]):
                    print("\n✅ Agent 已完成任务")
                    break
            
            # 如果达到最大迭代次数
            if iteration == max_iterations - 1:
                print("\n⚠️  达到最大迭代次数，停止执行")

        # 验证 Agent 是否遵循了技能指导
        final_response = response2.content
        skill_keywords = ["workflow", "步骤", "流程", "research", "研究", "sources", "来源", "搜索", "search"]
        if any(keyword in final_response.lower() for keyword in skill_keywords):
            print("\n✅ Agent 遵循了技能指导！")
        else:
            print("\n⚠️  Agent 可能没有完全遵循技能指导")

    print("\n" + "=" * 80)
    print("渐进式加载算法演示完成！")
    print("=" * 80)

    print("\n✅ 渐进式加载核心流程:")
    print("  1. ✓ 只加载元数据，不加载完整内容（节省 token）")
    print("  2. ✓ 技能列表注入系统提示（提高发现性）")
    print("  3. ✓ Agent 可以识别技能需求")
    print("  4. ✓ Agent 按需读取完整技能内容（延迟加载）")
    print("  5. ✓ Agent 可以按照技能指导执行任务")
    
    print(f"\n📁 技能文件位置:")
    print(f"  用户技能: {user_skills_dir}")
    print(f"  项目技能: {project_skills_dir}")
    print("\n💡 提示: 你可以直接编辑这些目录中的 SKILL.md 文件来测试不同的技能！")


if __name__ == "__main__":
    test_progressive_loading_with_llm()

