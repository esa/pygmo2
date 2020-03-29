// Copyright 2020 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/config.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include "bfe.hpp"
#include "common_utils.hpp"
#include "object_serialization.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

bfe_inner<py::object>::bfe_inner(const py::object &o)
{
    // Forbid the use of a pygmo.bfe as a UDBFE.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::bfe as a UDBFE is forbidden and prevented by the fact
    // that the generic constructor from UDBFE is disabled if the input
    // object is a pagmo::bfe (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a bfe, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input bfe.
    if (pygmo::type(o).equal(py::module::import("pygmo").attr("bfe"))) {
        pygmo::py_throw(PyExc_TypeError,
                        ("a pygmo.bfe cannot be used as a UDBFE for another pygmo.bfe (if you need to copy a "
                         "bfe please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "bfe");
    check_mandatory_method(o, "__call__", "bfe");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<bfe_inner_base> bfe_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<bfe_inner>(m_value);
}

vector_double bfe_inner<py::object>::operator()(const problem &p, const vector_double &dvs) const
{
    return pygmo::ndarr_to_vector<vector_double>(
        py::cast<py::array_t<double>>(m_value.attr("__call__")(p, pygmo::vector_to_ndarr<py::array_t<double>>(dvs))));
}

thread_safety bfe_inner<py::object>::get_thread_safety() const
{
    return thread_safety::none;
}

std::string bfe_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string bfe_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

#if PAGMO_VERSION_MAJOR > 2 || (PAGMO_VERSION_MAJOR == 2 && PAGMO_VERSION_MINOR >= 15)

std::type_index bfe_inner<py::object>::get_type_index() const
{
    return std::type_index(typeid(py::object));
}

const void *bfe_inner<py::object>::get_ptr() const
{
    return &m_value;
}

void *bfe_inner<py::object>::get_ptr()
{
    return &m_value;
}

#endif

template <typename Archive>
void bfe_inner<py::object>::save(Archive &ar, unsigned) const
{
    ar << boost::serialization::base_object<bfe_inner_base>(*this);
    ar << pygmo::object_to_vchar(m_value);
}

template <typename Archive>
void bfe_inner<py::object>::load(Archive &ar, unsigned)
{
    ar >> boost::serialization::base_object<bfe_inner_base>(*this);
    std::vector<char> v;
    ar >> v;
    m_value = pygmo::vchar_to_object(v);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_BFE_IMPLEMENT(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

// Serialization support for the bfe class.
py::tuple bfe_pickle_getstate(const pagmo::bfe &b)
{
    // The idea here is that first we extract a char array
    // into which b has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << b;
    }
    auto s = oss.str();
    return py::make_tuple(py::bytes(s.data(), boost::numeric_cast<py::size_t>(s.size())));
}

pagmo::bfe bfe_pickle_setstate(py::tuple state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialized the object.
    if (py::len(state) != 1) {
        pygmo::py_throw(PyExc_ValueError, ("the state tuple passed for bfe deserialization "
                                           "must have 1 element, but instead it has "
                                           + std::to_string(py::len(state)) + " element(s)")
                                              .c_str());
    }

    auto ptr = PyBytes_AsString(state[0].ptr());
    if (!ptr) {
        pygmo::py_throw(PyExc_TypeError, "a bytes object is needed to deserialize a bfe");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len(state[0])));
    pagmo::bfe b;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> b;
    }

    return b;
}

} // namespace pygmo